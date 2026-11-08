"""Regression checks execute actual route and nested-handler ASTs with synthetic IO.
No production modules with database/credential side effects are imported.
"""
from __future__ import annotations
import ast
import asyncio
import copy
import collections.abc
import importlib.util
import json
import logging
import os
import re
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import types
import uuid
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parents[1]
LIVE = True
from fastapi import HTTPException, Request, status

def source(path):
    return (ROOT / path).read_text()

def node(path, name):
    tree = ast.parse(source(path))
    return next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)

def load_function(path, name, ns, defaults=None):
    n=copy.deepcopy(node(path,name)); n.decorator_list=[]
    if defaults is not None: n.args.defaults=[ast.Constant(value=x) for x in defaults]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(ROOT/path),'exec'),ns)
    return ns[name]

# Keep the normal Python import paths for pure provider controls/security, while
# preventing subscriptions.__init__ from importing the live manager/database.
for name in ('open_webui','open_webui.subscriptions'):
    pkg=types.ModuleType(name); pkg.__path__=[]; sys.modules[name]=pkg
def load_pure(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod
controls=load_pure('open_webui.subscriptions.controls','backend/open_webui/subscriptions/controls.py')
security=load_pure('open_webui.subscriptions.security','backend/open_webui/subscriptions/security.py')
load_pure('open_webui.subscriptions.tasks','backend/open_webui/subscriptions/tasks.py')
routing=types.ModuleType('open_webui.subscriptions.routing')
load_function('backend/open_webui/subscriptions/routing.py','is_subscription_model',routing.__dict__)
sys.modules[routing.__name__]=routing
records={}
log=logging.getLogger('logic-repro'); logging.basicConfig(level=logging.ERROR)

async def repro_direct():
    def passed_guard(*_): raise RuntimeError('DIRECT_GUARD_PASSED')
    ns={'copy':copy,'HTTPException':HTTPException,'status':status,'log':log,
        '_set_direct_model':AsyncMock(),'Config':NS(get=AsyncMock(return_value={})), 'merge_model_params':passed_guard}
    f=load_function('backend/open_webui/main.py','chat_completion',ns,defaults=[None])
    request=NS(app=NS(state=NS(MODELS={'api':{'id':'api'}})),state=NS())
    try:
        await f(request,{'model':'direct-api','messages':[{'role':'user','content':'hello'}], 'model_item':{'id':'direct-api','direct':True}},NS(id='owner',role='admin'))
        raise AssertionError('Expected direct model regression')
    except HTTPException as exc:
        assert exc.status_code==400 and ('DIRECT_GUARD_PASSED' in exc.detail if LIVE else 'is_sub_model' in exc.detail)
        records['direct_connection']={'status':exc.status_code,'detail':exc.detail}

async def repro_saved():
    subid='subscription:claude:sonnet'
    payload={'model':subid,'messages':[],'params':{}}
    meta={'chat_id':'00000000-0000-4000-8000-000000000001','user_message':{'id':'u1','role':'user','content':'hello'},'user_message_id':'u1'}
    request=NS(state=NS(internal=True),app=NS(state=NS(redis=None)))
    # The real first-phase validation correctly accepts the UI request.
    security.validate_chat_request_security(payload,metadata=meta,request=request,require_complete_messages=False)
    reconstruct=AsyncMock(side_effect=RuntimeError('RECONSTRUCTION_REACHED'))
    ns={'copy':copy,'asyncio':asyncio,'HTTPException':HTTPException,'log':log,'time':time,
        'is_saved_chat_id':lambda _:True,'model_info':None,'model_id':subid,'model_controls':{},'is_new_chat':False,
        'apply_model_controls':lambda params,*args:params,'process_chat_payload':reconstruct}
    f=load_function('backend/open_webui/main.py','process_chat',ns)
    try:
        await f(request,payload,NS(id='owner'),meta,{'id':subid,'owned_by':'subscription'})
        raise AssertionError('Expected saved-chat rejection')
    except HTTPException as exc:
        assert not LIVE
        assert exc.status_code==400 and 'messages array cannot be empty' in exc.detail
        assert reconstruct.await_count==0
        records['saved_chat']={'first_phase':'accepted','second_phase_status':exc.status_code,'detail':exc.detail,'reconstruction_calls':reconstruct.await_count}
    except RuntimeError as exc:
        assert LIVE and str(exc)=='RECONSTRUCTION_REACHED' and reconstruct.await_count==1
        records['saved_chat']={'first_phase':'accepted','second_phase':'accepted','reconstruction_calls':reconstruct.await_count}

async def repro_native_reasoning_menu():
    subid = 'subscription:gemini:native-model-a'
    payload = {'model': subid, 'messages': [{'role': 'user', 'content': 'hello'}], 'params': {}}
    metadata = {'chat_id': None}
    request = NS(state=NS(internal=True), app=NS(state=NS(redis=None)))
    controls = {'reasoning_effort': {'label': 'Thinking', 'options': {
        'extra_high': {'label': 'Extra high', 'params': {'reasoning_effort': 'extra_high'}}}}}
    ns = {'deepcopy': copy.deepcopy, 'deep_update': lambda a, b: {**a, **b}, 'HTTPException': HTTPException}
    apply_controls = load_function('backend/open_webui/utils/payload.py', 'apply_model_controls', ns)
    async def middleware(request, body, *args, **kwargs):
        assert body['params']['custom_params']['reasoning_effort'] == 'extra_high'
        raise RuntimeError('NATIVE_REASONING_MENU_REACHED')
    ns = {'copy': copy, 'asyncio': asyncio, 'HTTPException': HTTPException, 'log': log, 'time': time,
        'is_saved_chat_id': lambda _: False, 'model_info': None, 'model_id': subid,
        'model_controls': {subid: {'reasoning_effort': 'extra_high'}}, 'is_new_chat': False,
        'apply_model_controls': apply_controls, 'process_chat_payload': middleware}
    process = load_function('backend/open_webui/main.py', 'process_chat', ns)
    try:
        await process(request, payload, NS(id='owner'), metadata,
            {'id': subid, 'owned_by': 'subscription', 'info': {'params': {'model_controls': controls}}})
        raise AssertionError('Native choice must reach middleware')
    except RuntimeError as exc:
        assert str(exc) == 'NATIVE_REASONING_MENU_REACHED'
    records['native_reasoning_menu'] = {'selected': 'extra_high', 'reached_middleware': True}

async def repro_post_filter_security_guard():
    fn = node('backend/open_webui/utils/middleware.py', 'process_chat_payload')
    setup = [copy.deepcopy(item) for item in fn.body if
        (isinstance(item, ast.ImportFrom) and item.module in ('open_webui.subscriptions.routing', 'open_webui.subscriptions.security'))
        or (isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'subscription_request' for target in item.targets))
        or (isinstance(item, ast.FunctionDef) and item.name == 'validate_subscription_effective_payload')]
    index = next(i for i,item in enumerate(fn.body) if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'features' for target in item.targets))
    guard_index = next(i for i in range(index-1, max(index-5, -1), -1) if ast.unparse(fn.body[i]) == 'validate_subscription_effective_payload()')
    wrapper = ast.parse('async def guarded_stage(request, form_data, metadata, model, user, extra_params):\n    pass').body[0]
    wrapper.body = setup + [ast.parse('validate_subscription_effective_payload()').body[0],
        ast.parse('form_data = await injected_inlet_filter(form_data)').body[0]] + copy.deepcopy(fn.body[guard_index:index+3])
    handlers = {'image': AsyncMock(), 'memory': AsyncMock()}
    for feature in ('image_generation', 'memory', 'web_search'):
        async def injected(body):
            return {**body, 'features': {feature: True}}
        ns = {'injected_inlet_filter': injected, 'Config': NS(get=AsyncMock(return_value=True)),
            'chat_image_generation_handler': handlers['image'], 'add_memory_context': handlers['memory']}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])), 'actual-post-filter-guard', 'exec'),ns)
        try:
            await ns['guarded_stage'](NS(state=NS(), app=NS(state=NS(MODELS={}))),
                {'model': 'subscription:gemini:native', 'messages': [{'role':'user','content':'hello'}]},
                {'features': {}, 'params': {'function_calling': 'legacy'}},
                {'id': 'subscription:gemini:native', 'owned_by': 'subscription'}, NS(id='owner',role='admin'), {})
            raise AssertionError('Injected feature reached handlers')
        except HTTPException as error:
            assert error.status_code == 400
    assert all(handler.await_count == 0 for handler in handlers.values())
    # A trusted filter can replace an API model with a non-prefixed alias.
    alias = {'id': 'team-claude', 'info': {'base_model_id': 'subscription:claude:sonnet'}}
    class Registry:
        # Match RedisDict's mapping interface without requiring a plain dict.
        def get(self, key):
            return alias if key == alias['id'] else None
    for registry in ({alias['id']: alias}, Registry()):
        async def injected(body):
            return {**body, 'model': alias['id'], 'features': {'image_generation': True}}
        ns = {'injected_inlet_filter': injected}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), 'actual-alias-retarget-guard', 'exec'), ns)
        try:
            await ns['guarded_stage'](NS(state=NS(), app=NS(state=NS(MODELS=registry))),
                {'model': 'api', 'messages': [{'role': 'user', 'content': 'hello'}]}, {},
                {'id': 'api', 'owned_by': 'openai'}, NS(id='owner', role='admin'), {})
            raise AssertionError('Retargeted alias reached host feature handlers')
        except HTTPException as error:
            assert error.status_code == 400
    records['post_filter_subscription_guard'] = {'forbidden_features_rejected': 3, 'host_handler_calls': 0}

async def repro_cache():
    owner=NS(id='owner'); other=NS(id='other')
    manager_mod=types.ModuleType('open_webui.subscriptions.manager')
    manager_mod.SAFE_SUBSCRIPTION_CAPABILITIES={'tools':False}
    nsmanager={'time':time,'copy':copy,'SAFE_SUBSCRIPTION_CAPABILITIES':manager_mod.SAFE_SUBSCRIPTION_CAPABILITIES}
    get_models=load_function('backend/open_webui/subscriptions/manager.py','get_models',nsmanager)
    cfg=NS(enabled=True)
    adapter=NS(execution_available=True,is_installed=lambda _: (True,'fake-claude'),get_supported_models=lambda _: [{'id':'sonnet','name':'Sonnet'}])
    manager=NS(_adapters={'claude':adapter},get_runtime_config=AsyncMock(return_value=(True,'owner',60,{'claude':cfg})))
    async def bound_get_models(user=None): return await get_models(manager,user=user)
    manager_mod.subscription_manager=NS(get_models=bound_get_models); sys.modules[manager_mod.__name__]=manager_mod
    config={'openai.enable':True,'ollama.enable':False,'subscriptions.enable':True,'models.base_models_cache':True,
            'evaluation.arena.enable':False,'models.default_metadata':{},'models.default_params':{}}
    async def conf(*keys): return {k:config.get(k) for k in keys}
    ns={'asyncio':asyncio,'copy':copy,'Config':NS(get_many=conf),'ENABLE_FUNCTIONS':False,'log':log,
        'Functions':NS(get_functions_by_ids=AsyncMock(return_value=[]),get_function_valves_by_ids=AsyncMock(return_value={})),
        'Models':NS(get_all_models=AsyncMock(return_value=[])),'get_functions_cache':lambda _: {},'RedisDict':type('RedisDict',(),{}),
        'get_function_models':AsyncMock(return_value=[]),'fetch_openai_models':AsyncMock(return_value=[{'id':'api','owned_by':'openai'}]),
        'fetch_ollama_models':AsyncMock(return_value=[])}
    for name in ('fetch_subscription_models','get_all_base_models','get_all_models'):
        load_function('backend/open_webui/utils/models.py',name,ns)
    req=NS(app=NS(state=NS(redis=None,MODELS={},BASE_MODELS=[])))
    first=await ns['get_all_models'](req,user=other)
    second=await ns['get_all_models'](req,user=owner)
    canonical=await bound_get_models(user=None)
    assert any(m['id'].startswith('subscription:') for m in canonical)
    assert any(m['id'].startswith('subscription:') for m in second) is LIVE
    records['shared_catalog']={'first_nonowner':[m['id'] for m in first], 'then_owner':[m['id'] for m in second], 'canonical':[m['id'] for m in canonical]}
    adapter.get_supported_models = lambda _: [{'id':'opus','name':'Updated Native Catalog'}]
    refreshed = await ns['get_all_models'](req,user=owner)
    assert 'subscription:claude:opus' in {model['id'] for model in refreshed}
    assert 'subscription:claude:sonnet' not in {model['id'] for model in refreshed}
    assert ns['fetch_openai_models'].await_count == 1
    records['shared_catalog']['fresh_subscription_over_cached_api'] = True

async def repro_stream_scope():
    # Compile the exact original nested function within a small synthetic closure.
    # Its body and nonlocal declarations are unchanged.
    stream_fn=copy.deepcopy(node('backend/open_webui/utils/middleware.py','stream_body_handler'))
    wrapper=ast.parse('async def harness(response):\n    usage = {}\n    current_usage = {}\n    finish_reason = None\n    output = []\n    prior_output = []\n    last_response_id = None\n    stream_size = 0\n    stream_failed = False\n    stream_error_payload = None\n').body[0]
    wrapper.body += [stream_fn] + ast.parse('await stream_body_handler(response, form_data)\nreturn {"outer_failed": stream_failed, "outer_error": stream_error_payload, "inner_locals": stream_body_handler.__code__.co_varnames, "inner_freevars": stream_body_handler.__code__.co_freevars}').body
    events=[]
    async def event_emitter(event): events.append(event)
    async def chunks(): yield 'data: '+json.dumps({'error':{'type':'subscription_error','message':'CLI_STREAM_ERROR'}})+'\n\n'
    ns={'asyncio':asyncio,'log':log,'JSONCodec':NS(loads=json.loads),'form_data':{'model':'subscription:claude:sonnet'},
        'CHAT_RESPONSE_STREAM_DELTA_CHUNK_SIZE':1,'metadata':{},'filter_functions':[], 'save_to_chat':False,
        'event_emitter':event_emitter,'save_current_response_stream':AsyncMock(),'normalize_usage':lambda v:v,
        'request':NS(state=NS(),app=NS(state=NS(redis=None))),'tool_calls':[]}
    module=ast.Module(body=[wrapper],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'actual-stream-handler-fixture','exec'),ns)
    result=await ns['harness'](NS(body_iterator=chunks()))
    assert result['outer_failed'] is LIVE
    assert ('stream_failed' in result['inner_freevars']) is LIVE
    assert events[0]['data'].get('error')
    records['stream_failure_scope']={'outer_failed':result['outer_failed'],'outer_error':result['outer_error'],
        'inner_has_local_failure_flag':'stream_failed' in result['inner_locals'],'error_was_emitted':True,'outer_success_gate_taken':not result['outer_failed']}

async def repro_arena():
    targets=['subscription:claude:sonnet','subscription:codex:default']
    choices=[]; captured=[]
    def draw(candidates):
        chosen=targets[len(choices)]; assert chosen in candidates; choices.append(chosen); return chosen
    routing.__dict__.update(random=NS(choice=draw), is_subscription_owner=AsyncMock(return_value=True), HTTPException=HTTPException, status=status)
    load_function('backend/open_webui/subscriptions/routing.py','resolve_arena_model',routing.__dict__)
    modelmap={
        'arena':{'id':'arena','owned_by':'arena','info':{'meta':{'model_ids':targets}}},
        **{mid:{'id':mid,'owned_by':'subscription'} for mid in targets}}
    async def config_get(key,*defaults): return {} if key=='models.default_params' else False
    async def create_task(redis,process,**kwargs):
        await process
        return 'synthetic-task',None
    ns={'copy':copy,'asyncio':asyncio,'HTTPException':HTTPException,'status':status,'log':log,
        'Models':NS(get_model_by_id=AsyncMock(return_value=None)), 'Config':NS(get=config_get),
        'BYPASS_MODEL_ACCESS_CONTROL':False,'BYPASS_ADMIN_ACCESS_CONTROL':False,
        'check_model_access':AsyncMock(),'merge_model_params':lambda a,b:{**a,**b},
        'is_saved_chat_id':lambda _:False,'is_temporary_chat_id':lambda _:False,
        'normalize_chat_variables':lambda x:x,'uuid4':uuid.uuid4,'create_task':create_task,
        'get_event_emitter':AsyncMock(return_value=None),'captured':captured}
    fnode=copy.deepcopy(node('backend/open_webui/main.py','chat_completion'))
    fnode.decorator_list=[]; fnode.args.defaults=[ast.Constant(None)]
    replacement=ast.parse('async def process_chat(request, form_data, user, metadata, model, tasks=None):\n    captured.append({"executed_model":form_data["model"], "model_dict":model["id"], "selected_model_id":metadata.get("selected_model_id"), "state_metadata_is_own":request.state.metadata is metadata})\n    return {}').body[0]
    class ReplaceWorker(ast.NodeTransformer):
        def visit_AsyncFunctionDef(self,n):
            if n.name=='process_chat': return replacement
            return self.generic_visit(n)
    fnode=ReplaceWorker().visit(fnode)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fnode],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'actual-outer-chat-routing-fixture','exec'),ns)
    req=Request({'type':'http','app':NS(state=NS(MODELS=modelmap,redis=None)),'state':{},'headers':[]})
    payload={'model':'arena','messages':[{'role':'user','content':'hello'}],
        'message_ids':[{'model_id':'arena','message_id':'assistant-fixture'}],
        'chat_id':'fixture-chat','session_id':'fixture-session','user_message':{'id':'u1','role':'user','content':'hello'}}
    await ns['chat_completion'](req,payload,NS(id='owner',role='admin',name='Owner'))
    if LIVE:
        assert len(choices)==1 and captured[0]['executed_model']==targets[0] and captured[0]['selected_model_id']==targets[0]
    else:
        assert len(choices)==2 and captured[0]['executed_model']==targets[1] and captured[0]['selected_model_id'] is None
    records['arena_selection']={'draws':list(choices),'worker':copy.deepcopy(captured[0])}
    if LIVE:
        choices.clear(); captured.clear()
        two={'model':'arena','messages':[{'role':'user','content':'hello'}],
            'message_ids':[{'model_id':'arena','message_id':'a1'},{'model_id':'arena','message_id':'a2'}],
            'chat_id':'fixture-chat','session_id':'fixture-session','user_message':{'id':'u1','role':'user','content':'hello'}}
        await ns['chat_completion'](req,two,NS(id='owner',role='admin',name='Owner'))
        assert len(choices)==2 and len(captured)==2
        assert all(c['state_metadata_is_own'] and c['executed_model']==c['selected_model_id'] for c in captured)
        records['arena_two_columns']={'draws':list(choices),'workers':copy.deepcopy(captured)}
    modelmap['private-arena']={'id':'private-arena','owned_by':'arena','info':{'meta':{'model_ids':targets,'access_grants':[{'principal_id':'somebody-else'}]}}}
    access_calls=[]
    async def gated_access(user,model,**kwargs):
        access_calls.append(model['id'])
        if model['id']=='private-arena': raise HTTPException(403,'PRIVATE_ARENA_DENIED')
    ns['check_model_access']=gated_access
    for label,primary,entries in [
        ('primary','private-arena',[{'model_id':'private-arena','message_id':'private-placeholder'}]),
        ('secondary',targets[0],[{'model_id':targets[0],'message_id':'public-placeholder'},{'model_id':'private-arena','message_id':'private-placeholder'}]),
    ]:
        access_calls.clear(); choices.clear(); captured.clear()
        new_payload={'model':primary,'messages':[{'role':'user','content':'hello'}],'message_ids':entries,
            'chat_id':'fixture-chat','session_id':'fixture-session','user_message':{'id':'u1','role':'user','content':'hello'}}
        outcome='executed'
        try:
            await ns['chat_completion'](req,new_payload,NS(id='owner',role='user',name='Owner'))
        except HTTPException as exc:
            assert exc.status_code==403 and exc.detail=='PRIVATE_ARENA_DENIED'
            outcome='denied'
        assert outcome == 'denied' and not choices and not captured
        records['arena_acl_'+label]={'outcome':outcome,'access_checked':list(access_calls),'draws':list(choices),'worker_count':len(captured)}

async def repro_failed_event():
    tree=ast.parse(source('backend/open_webui/utils/middleware.py'))
    gate=copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='stream_failed'))
    fn=ast.AsyncFunctionDef(name='failure_gate',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[gate],decorator_list=[])
    failed_event=AsyncMock(); finished_event=AsyncMock(); events=[]
    async def emit(event): events.append(event)
    ns={'stream_failed':True,'stream_error_payload':{'message':'CLI_STREAM_ERROR'},'asyncio':asyncio,'time':time,
        'form_data':{'model':'subscription:claude:sonnet'},'model':{'id':'subscription:claude:sonnet','owned_by':'subscription'},
        'response':NS(body_iterator=NS(aclose=AsyncMock())),'full_output':lambda:[],'save_to_chat':True,
        'metadata':{'chat_id':'00000000-0000-4000-8000-000000000001','message_id':'a1'},
        'Chats':NS(upsert_message_to_chat_by_id_and_message_id=AsyncMock()),'request':NS(state=NS(internal=False),app=NS(state=NS(redis=None))),
        'clear_response_stream':AsyncMock(),'response_stream_task_id':'task1','event_emitter':emit,
        'publish_chat_failed_event':failed_event,'publish_chat_finished_event':finished_event,'user':NS(id='owner')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual-failure-gate-fixture','exec'),ns)
    await ns['failure_gate']()
    assert failed_event.await_count==(1 if LIVE else 0) and finished_event.await_count==0
    records['failure_event']={'failure_events':failed_event.await_count,'success_events':finished_event.await_count,'ui_error_events':len(events)}

async def repro_bulk_model_owner_guards():
    """Both denied bulk routes must be atomic, while the owner can import/sync."""
    class FieldObject(NS):
        def model_dump(self):
            return {}

    class Form:
        def __init__(self, **data):
            self.__dict__.update(data)
            self.params = FieldObject(model_fields_set=set())
            self.meta = FieldObject(background_image_url=None, voice_avatar=None, model_fields_set=set())
            self.access_grants = []

        def model_dump(self):
            return vars(self)

    class FakeModels:
        writes = []
        existing = []

        @classmethod
        async def get_models_by_ids(cls, *args, **kwargs):
            return cls.existing

        @classmethod
        async def get_all_models(cls, **kwargs):
            return cls.existing

        @classmethod
        async def insert_new_model(cls, user_id, form_data, db):
            cls.writes.append(form_data.id)
            return form_data

        @classmethod
        async def sync_models(cls, user_id, models, db):
            cls.writes.append('sync')
            return models

    noop = AsyncMock(return_value=None)
    ns = {'Models': FakeModels, 'Config': NS(get=AsyncMock(return_value={})), 'ModelForm': Form,
          'HTTPException': HTTPException, 'status': status, 'is_valid_model_id': lambda value: True,
          '_verify_knowledge_file_access': noop, '_verify_background_image': noop, '_verify_voice_avatar': noop,
          'filter_allowed_access_grants': AsyncMock(return_value=[]), 'publish_event': noop,
          'EVENTS': NS(MODEL_IMPORTED='import', MODEL_SYNCED='sync'), 'log': log}
    for name in ('_references_subscription_model', '_check_subscription_model_config_access', '_check_model_controls', 'import_models', 'sync_models'):
        defaults = [None, None, None] if name == 'import_models' else [None, None] if name == 'sync_models' else None
        load_function('backend/open_webui/routers/models.py', name, ns, defaults=defaults)
    request = NS(app=NS(state=NS(MODELS={})))
    sub = {'id': 'subscription-alias', 'base_model_id': 'subscription:codex:default', 'name': 'Alias', 'params': {}, 'meta': {}}
    api = {**sub, 'id': 'api-alias', 'base_model_id': 'api'}
    routing.is_subscription_owner = AsyncMock(return_value=False)
    user = NS(id='other-admin', role='admin')
    for route in ('import', 'sync', 'sync-delete'):
        FakeModels.writes.clear()
        FakeModels.existing = [Form(**sub)] if route == 'sync-delete' else []
        try:
            if route == 'import':
                await ns['import_models'](request, user, NS(models=[copy.deepcopy(api), copy.deepcopy(sub)]), None)
            else:
                await ns['sync_models'](request, NS(models=[] if route == 'sync-delete' else [Form(**sub)]), user, None)
            raise AssertionError(f'{route} must reject a non-owner')
        except HTTPException as exc:
            assert exc.status_code == 403 and not FakeModels.writes
    routing.is_subscription_owner = AsyncMock(return_value=True)
    FakeModels.existing = []
    assert await ns['import_models'](request, NS(id='owner', role='admin'), NS(models=[copy.deepcopy(sub)]), None)
    assert await ns['sync_models'](request, NS(models=[Form(**sub)]), NS(id='owner', role='admin'), None)
    assert FakeModels.writes == ['subscription-alias', 'sync']
    records['bulk_model_owner_guards'] = {'nonowner_rejected_atomically': 3, 'owner_writes': 2}

async def repro_compaction_subscription_defaults():
    chat_module = types.ModuleType('open_webui.utils.chat')
    sys.modules[chat_module.__name__] = chat_module
    settings = {'task.model.params': {}, 'chat.context_compaction.model': None}
    ns = {'copy': copy, 'collections': collections, 'Config': NS(get_many=AsyncMock(side_effect=lambda *keys: settings)),
          'JSONCodec': NS(loads=json.loads, JSONDecodeError=json.JSONDecodeError),
          'DEFAULT_CONTEXT_COMPACTION_PROMPT': 'Summarize', 'log': log,
          'get_content_from_message': lambda m: m.get('content', ''),
          'get_last_user_message': lambda messages: messages[-1].get('content', ''),
          'replace_prompt_variable': lambda text, *args: text,
          'replace_messages_variable': lambda text, *args: text,
          'prompt_variables_template': lambda text, *args: text,
          'prompt_template': AsyncMock(side_effect=lambda text, user: text),
          '_response_text': lambda response: response or ''}
    load_function('backend/open_webui/utils/misc.py', 'deep_update', ns)
    load_function('backend/open_webui/utils/payload.py', 'apply_params_to_form_data', ns)
    for name in ('_message_prefix', '_generate_summary'):
        load_function('backend/open_webui/utils/context_compaction.py', name, ns)
    ns['is_subscription_model'] = routing.is_subscription_model
    load_function('backend/open_webui/routers/tasks.py', 'apply_task_model_params', ns)
    source_metadata = {'tool_ids': ['source-tool'], 'features': {'web_search': True}, 'chat_id': 'source-chat'}
    request = NS(state=NS(metadata=copy.deepcopy(source_metadata), terminal_id='source-terminal'))
    previous = [{'role': 'user', 'content': 'old turn'}]
    recent = [{'role': 'user', 'content': 'recent turn'}]
    configured = {'max_tokens': 50, 'temperature': .2,
                  'custom_params': {'options': '{"max_completion_tokens":42,"reasoning_effort":"high"}'}}
    for mid in ('subscription:claude:sonnet', 'subscription:codex:default',
                'subscription:gemini:native-a', 'workspace-preset', 'api', 'arena-sub', 'arena-api'):
        model = {'id': mid, 'owned_by': 'openai'}
        if mid == 'workspace-preset':
            model['info'] = {'base_model_id': 'subscription:claude:sonnet'}
        models = {mid: model}
        native = mid not in ('api', 'arena-api')
        if mid.startswith('arena-'):
            chosen = 'subscription:claude:sonnet' if native else 'api'
            model.update(owned_by='arena', info={'meta': {'model_ids': [chosen]}})
            models[chosen] = {'id': chosen, 'owned_by': 'subscription' if native else 'openai'}
            routing.random = NS(choice=lambda candidates: candidates[0])
        else:
            task_payload = ns['apply_task_model_params']({'model': mid}, models, mid, copy.deepcopy(configured))
            if native:
                controls.validate_subscription_chat_controls(task_payload)
            else:
                assert task_payload['max_tokens'] == 50
        request.app = NS(state=NS(MODELS=models))
        for defaults in ({}, configured):
            for replay in (False, True):
                settings['task.model.params'] = copy.deepcopy(defaults)
                async def generate(request, form_data, **kwargs):
                    if native:
                        controls.validate_subscription_chat_controls(form_data)
                        security.validate_chat_request_security(form_data, metadata=request.state.metadata, request=request)
                        assert 'max_tokens' not in form_data
                    else:
                        assert form_data['max_tokens'] == (50 if defaults else 1000)
                        assert request.state.metadata['tool_ids'] == ['source-tool']
                    return 'verified summary'
                chat_module.generate_chat_completion = AsyncMock(side_effect=generate)
                body = {'messages': previous + recent} if replay else None
                summary = await ns['_generate_summary'](request, body, NS(id='owner'), mid,
                    models, previous, recent, None, 'Summarize')
                assert summary == 'verified summary'
                assert chat_module.generate_chat_completion.await_count == 1
    assert configured['custom_params']['options'].startswith('{')  # server config was not mutated
    assert request.state.metadata == source_metadata
    records['compaction_subscription_defaults'] = {'providers_and_alias': 4, 'branches': 2, 'api_defaults_retained': True}


async def repro_custom_subscription_menus():
    native_id = 'subscription:gemini:native-a'
    native_controls = {'reasoning_effort': {'label': 'Thinking', 'options': {
        'high': {'label': 'High', 'params': {'reasoning_effort': 'high'}}}}}
    native = {'id': native_id, 'name': 'Native', 'owned_by': 'subscription',
              'info': {'params': {'model_controls': native_controls}}}
    config = {'subscriptions.enable': True, 'models.base_models_cache': False,
              'evaluation.arena.enable': False, 'models.default_metadata': {}, 'models.default_params': {}}
    ns = {'asyncio': asyncio, 'copy': copy, 'log': log, 'ENABLE_FUNCTIONS': False,
          'Config': NS(get_many=AsyncMock(side_effect=lambda *keys: {k: config.get(k) for k in keys})),
          'Models': NS(get_all_models=AsyncMock()),
          'Functions': NS(get_functions_by_ids=AsyncMock(return_value=[]), get_function_valves_by_ids=AsyncMock(return_value={})),
          'get_functions_cache': lambda _: {}, 'RedisDict': type('RedisDict', (), {}),
          'get_chat_variables_schema': lambda _: None,
          'get_all_base_models': AsyncMock(side_effect=lambda *a, **kw: [copy.deepcopy(native), {'id': 'api', 'owned_by': 'openai'}])}
    load_function('backend/open_webui/utils/models.py', 'public_model_params', ns)
    get_models = load_function('backend/open_webui/utils/models.py', 'get_all_models', ns)
    def saved(mid, base, params=None):
        data = {'id': mid, 'name': 'My preset', 'base_model_id': base, 'params': params or {}, 'meta': {}}
        return NS(id=mid, name='My preset', base_model_id=base, is_active=True, created_at=0,
                  meta=None, params=NS(model_dump=lambda: copy.deepcopy(data['params'])),
                  model_dump=lambda: copy.deepcopy(data))
    for rows in ([saved(native_id, None), saved('preset', native_id)],
                 [saved('preset', native_id), saved(native_id, None)]):
        ns['Models'].get_all_models.return_value = rows + [saved('api-preset', 'api')]
        request = NS(app=NS(state=NS(redis=None, MODELS={}, BASE_MODELS=[])))
        models = {m['id']: m for m in await get_models(request)}
        for mid in (native_id, 'preset'):
            assert models[mid]['info']['params']['model_controls'] == native_controls
            # The internal registry must retain the actual native parameter mapping.
            assert request.app.state.MODELS[mid]['info']['params']['model_controls']['reasoning_effort']['options']['high']['params'] == {'reasoning_effort': 'high'}
        assert models['api-preset']['info']['params'] == {}
    override = {'reasoning_effort': {'label': 'Custom thinking', 'options': {'custom': {'label': 'Custom', 'params': {'private': 'internal-only'}}}}}
    ns['Models'].get_all_models.return_value = [saved('preset', native_id, {'model_controls': override, 'private': 'hidden'})]
    models = {m['id']: m for m in await get_models(NS(app=NS(state=NS(redis=None, MODELS={}, BASE_MODELS=[]))))}
    published = models['preset']['info']['params']
    assert published['model_controls']['reasoning_effort']['label'] == 'Custom thinking'
    assert published['model_controls']['reasoning_effort']['options']['custom'] == {'label': 'Custom'}
    assert 'private' not in published
    assert native['info']['params']['model_controls'] == native_controls
    records['custom_subscription_menus'] = {'base_and_alias_inherit_native_mappings': True, 'custom_priority_and_redaction': True}


async def repro_subscription_log_privacy():
    logger = logging.getLogger('subscription-log-regression')
    messages = []
    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())
    handler = Capture(); logger.addHandler(handler)
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.propagate = False
    manager = sys.modules['open_webui.subscriptions.manager'].subscription_manager
    manager.generate_chat_completion = AsyncMock(return_value={'ok': True})
    from fastapi.responses import StreamingResponse
    ns = {'log': logger, 'StreamingResponse': StreamingResponse,
          'BYPASS_MODEL_ACCESS_CONTROL': False, 'check_model_access': AsyncMock(),
          'generate_openai_chat_completion': AsyncMock(return_value={'ok': True})}
    generate = load_function('backend/open_webui/utils/chat.py', 'generate_chat_completion', ns)
    base = 'subscription:claude:sonnet'
    registry = {base: {'id': base, 'owned_by': 'subscription'},
                'preset': {'id': 'preset', 'info': {'base_model_id': base}},
                'api': {'id': 'api', 'owned_by': 'openai'}}
    registry['arena-sub'] = {'id': 'arena-sub', 'owned_by': 'arena', 'info': {'meta': {'model_ids': [base]}}}
    registry['arena-api'] = {'id': 'arena-api', 'owned_by': 'arena', 'info': {'meta': {'model_ids': ['api']}}}
    routing.random = NS(choice=lambda candidates: candidates[0])
    request = NS(state=NS(), app=NS(state=NS(MODELS=registry)))
    try:
        for level in (logging.DEBUG, logging.INFO):
            logger.setLevel(level)
            for mid in registry:
                messages.clear()
                payload = {'model': mid, 'messages': [{'role': 'system', 'content': 'synthetic system instructions'},
                    {'role': 'user', 'content': 'synthetic private prompt'}], 'stream': False}
                await generate(request, payload, NS(id='owner', role='admin'), bypass_system_prompt=True)
                logged = '\n'.join(messages)
                should_log = mid in ('api', 'arena-api') and level == logging.DEBUG
                assert ('synthetic private prompt' in logged) == should_log
                assert ('synthetic system instructions' in logged) == should_log
    finally:
        logger.removeHandler(handler); logger.setLevel(previous_level); logger.propagate = previous_propagate
    records['subscription_log_privacy'] = {'subscription_and_alias_redacted': True, 'api_debug_retained': True}


async def repro_saved_history_before_media_io():
    mid = 'subscription:claude:sonnet'
    previous = {'id': 'old', 'role': 'user', 'content': [{'type': 'text', 'text': 'Describe'},
        {'type': 'image_url', 'image_url': {'url': 'https://history.example.invalid/private.png'}}]}
    current = {'id': 'new', 'role': 'user', 'content': 'Now answer as text'}
    for target, with_image in ((mid, True), ('saved-preset', True), ('api', True), (mid, False)):
        model = {'id': target, 'owned_by': 'openai'}
        if target == 'saved-preset':
            model['info'] = {'base_model_id': mid}
        image_io = AsyncMock(return_value='data:image/png;base64,ZmFrZQ==')
        ns = {'log': log, 'UserModel': type('UserModel', (), {}), 'is_saved_chat_id': lambda _: True,
              'is_temporary_chat_id': lambda _: False, 'apply_params_to_form_data': lambda body, _: body,
              'Chats': NS(get_chat_by_id=AsyncMock(return_value=NS(user_id='owner', meta={})), get_chat_folder_id=AsyncMock(return_value=None)),
              'load_messages_from_db': AsyncMock(return_value=[copy.deepcopy(previous), current] if with_image else [current]),
              'is_raster_image_content_type': lambda _: False, 'get_system_message': lambda _: None,
              'ENABLE_FUNCTIONS': False, 'get_sorted_filters': lambda *_: [],
              'apply_summary_checkpoint': AsyncMock(side_effect=lambda messages: (messages, None)),
              'process_messages_with_output': lambda messages, **_: messages, 'get_reasoning_format': lambda _: None,
              'sanitize_tool_pairs': lambda messages: messages, 'get_image_base64_from_url': image_io,
              'get_event_emitter': AsyncMock(side_effect=RuntimeError('MEDIA_STAGE_REACHED'))}
        load_function('backend/open_webui/utils/middleware.py', 'convert_url_images_to_base64', ns)
        middleware = load_function('backend/open_webui/utils/middleware.py', 'process_chat_payload', ns)
        request = NS(state=NS(), app=NS(state=NS(MODELS={target: model})))
        metadata = {'chat_id': '00000000-0000-4000-8000-000000000001', 'user_message_id': 'new',
                    'user_message': current, 'params': {'function_calling': 'native'}}
        try:
            await middleware(request, {'model': target, 'messages': [], 'params': {}}, NS(id='owner', role='admin'), metadata, model)
            raise AssertionError('Expected rejection or end of synthetic IO boundary')
        except HTTPException as exc:
            assert target != 'api' and with_image and exc.status_code == 400
            assert image_io.await_count == 0
        except RuntimeError as exc:
            assert str(exc) == 'MEDIA_STAGE_REACHED'
            assert target == 'api' or not with_image
            assert image_io.await_count == int(with_image)
    records['saved_history_media'] = {'subscription_image_io_before_rejection': 0, 'text_and_api_media_preserved': True}


async def repro_model_mutation_owner_gates():
    class Row(NS):
        def model_dump(self):
            return dict(vars(self))
    base = 'subscription:claude:sonnet'
    sub = Row(id=base, user_id='owner', base_model_id=None, name='Claude', is_active=True)
    alias = Row(id='private-preset', user_id='owner', base_model_id=base, name='Preset', is_active=True)
    api = Row(id='api', user_id='owner', base_model_id=None, name='API', is_active=True)
    rows = {row.id: row for row in (sub, alias, api)}
    writes = []
    async def toggle(mid, **kw):
        writes.append(('toggle', mid)); return rows[mid]
    async def grants(*args, **kw):
        writes.append(('grants', args[1]))
    async def rewrite(mid, form, **kw):
        writes.append(('version', mid))
        rows[mid].version_id = kw['production_version_id']
        return rows[mid]
    models = NS(get_model_by_id=AsyncMock(side_effect=lambda mid, **kw: rows.get(mid)),
                toggle_model_by_id=toggle, insert_new_model=AsyncMock(), update_model_by_id=rewrite,
                update_model_updated_at_by_id=AsyncMock())
    class Form(Row):
        def __init__(self, **data):
            super().__init__(**data)
            self.params = NS(model_fields_set=set())
    ns = {'Models': models, 'HTTPException': HTTPException, 'status': status,
          'AccessGrants': NS(has_access=AsyncMock(return_value=False), set_access_grants=grants),
          'publish_event': AsyncMock(), 'model_response': lambda row: row.model_dump(),
          'EVENTS': NS(MODEL_ENABLED='enable', MODEL_DISABLED='disable', MODEL_ACCESS_UPDATED='access', MODEL_UPDATED='update'),
          'Config': NS(get=AsyncMock(return_value={})), 'validate_access_grants': AsyncMock(return_value=[]),
          'authorized_model_history': AsyncMock(return_value=alias),
          'ModelHistories': NS(get_history_by_id=AsyncMock(return_value=NS(id='old', snapshot={
              'base_model_id': 'api', 'name': 'Old API preset', 'meta': NS(background_image_url=None, voice_avatar=None)}))),
          'ModelForm': Form, 'deepcopy': copy.deepcopy, '_check_model_controls': AsyncMock(),
          '_verify_version_dependencies': AsyncMock(), '_verify_background_image': AsyncMock(), '_verify_voice_avatar': AsyncMock()}
    for name in ('_references_subscription_model', '_check_subscription_model_config_access', 'toggle_model_by_id', 'update_model_access_by_id', 'set_model_version'):
        load_function('backend/open_webui/routers/models.py', name, ns, defaults=None if name in ('_references_subscription_model', '_check_subscription_model_config_access') else [])
    request = NS(app=NS(state=NS(MODELS={mid: row.model_dump() for mid, row in rows.items()})))
    original_owner = routing.is_subscription_owner
    try:
        routing.is_subscription_owner = AsyncMock(return_value=False)
        for mid in (base, alias.id, 'subscription:codex:default'):
            if mid in rows:
                try:
                    await ns['toggle_model_by_id'](request, mid, NS(id='other', role='admin'), None)
                    raise AssertionError('Non-owner toggled subscription model')
                except HTTPException as exc:
                    assert exc.status_code == 403
            try:
                await ns['update_model_access_by_id'](request, NS(id=mid, name=None, access_grants=[]), NS(id='other', role='admin'), None)
                raise AssertionError('Non-owner changed subscription model grants')
            except HTTPException as exc:
                assert exc.status_code == 403
        try:
            await ns['set_model_version'](request, alias.id, NS(version_id='old'), NS(id='other', role='admin'), None)
            raise AssertionError('Old API snapshot replaced current subscription preset')
        except HTTPException as exc:
            assert exc.status_code == 403
        assert writes == [] and models.insert_new_model.await_count == 0
        version_lookup = ns['ModelHistories'].get_history_by_id
        # Exercise the real management readers with synthetic storage. Their
        # generic administrator bypass must not reveal subscription presets.
        for row in rows.values():
            row.meta = NS(background_image_url=None)
            row.params = {'system': 'synthetic-owner-private' if row.id != 'api' else 'api-public'}
        history_sub = Row(id='old-sub', model_id='api', snapshot={'base_model_id': base, 'params': {'system': 'synthetic-owner-private'}})
        history_api = Row(id='old-api', model_id='api', snapshot={'base_model_id': 'api', 'params': {}})
        async def searched(*args, filter, **kw):
            items = [row for row in rows.values() if not filter.get('exclude_subscription_models') or not ns['_references_subscription_model'](request, row)]
            return NS(items=items, total=len(items))
        models.search_models = searched
        models.get_all_models = AsyncMock(side_effect=lambda **kw: list(rows.values()))
        models.get_base_models = models.get_all_models
        models.get_models = AsyncMock(side_effect=lambda ids=None, **kw: [row for row in rows.values() if ids is None or row.id in ids])
        ns.update(BYPASS_ADMIN_ACCESS_CONTROL=True, PAGE_ITEM_COUNT=30,
            Groups=NS(get_groups_by_member_id=AsyncMock(return_value=[])),
            ModelAccessResponse=dict, ModelAccessListResponse=dict,
            add_chat_variables_schema=lambda value: value,
            model_response=lambda row: {'id': row.id, 'name': row.name, 'params': row.params, 'meta': {}, 'base_model_id': row.base_model_id},
            ERROR_MESSAGES=NS(NOT_FOUND='missing', ACCESS_PROHIBITED='denied', UNAUTHORIZED='denied'))
        ns['AccessGrants'].get_accessible_resource_ids = AsyncMock(return_value=set())
        ns['ModelHistories'].get_history_by_model_id = AsyncMock(return_value=[history_sub, history_api])
        ns['ModelHistories'].get_history_by_id = AsyncMock(side_effect=lambda mid, hid, **kw: history_sub if hid == 'old-sub' else history_api)
        ns['ModelHistories'].delete_history_entry = AsyncMock(return_value=True)
        for name in ('_filter_subscription_model_records', 'authorized_model_history', 'get_models', 'get_all_model_records',
                     'get_base_models', 'get_model_by_id', 'export_models', 'get_model_history', 'get_model_history_entry', 'delete_model_history_entry'):
            load_function('backend/open_webui/routers/models.py', name, ns,
                defaults=None if name in ('_filter_subscription_model_records', 'authorized_model_history') else [])
        other = NS(id='other', role='admin')
        for name in ('get_all_model_records', 'get_base_models'):
            kwargs = {'tag': None} if name == 'get_base_models' else {}
            visible = await ns[name](request=request, user=other, db=None, **kwargs)
            assert [row.id for row in visible] == ['api']
        visible = await ns['get_models'](request=request, user=other, db=None, query=None, view_option=None,
            tag=None, order_by=None, direction=None, page=1)
        assert [row['id'] for row in visible['items']] == ['api'] and visible['total'] == 1
        exported = await ns['export_models'](request, None, other, None)
        assert [row['id'] for row in exported] == ['api']
        assert 'synthetic-owner-private' not in str(exported)
        for name, kwargs in (
            ('get_model_by_id', {'id': alias.id}), ('get_model_history', {'id': alias.id, 'page': 1}),
            ('get_model_history_entry', {'id': 'api', 'history_id': 'old-sub'}),
            ('delete_model_history_entry', {'id': 'api', 'history_id': 'old-sub'}),
            ('export_models', {'ids': [alias.id]})):
            try:
                await ns[name](request=request, user=other, db=None, **kwargs)
                raise AssertionError(f'{name} exposed subscription config/history')
            except HTTPException as exc:
                assert exc.status_code == 403
        assert ns['ModelHistories'].delete_history_entry.await_count == 0
        visible_history = await ns['get_model_history'](request, 'api', 1, other, None)
        assert [entry.id for entry in visible_history] == ['old-api']
        await ns['toggle_model_by_id'](request, 'api', NS(id='other', role='admin'), None)
        await ns['update_model_access_by_id'](request, NS(id='api', access_grants=[]), NS(id='other', role='admin'), None)
        routing.is_subscription_owner = AsyncMock(return_value=True)
        assert len(await ns['get_all_model_records'](request, NS(id='owner', role='admin'), None)) == 3
        ns['ModelHistories'].get_history_by_id = version_lookup
        await ns['toggle_model_by_id'](request, base, NS(id='owner', role='admin'), None)
        await ns['set_model_version'](request, alias.id, NS(version_id='old'), NS(id='owner', role='admin'), None)
        assert writes == [('toggle', 'api'), ('grants', 'api'), ('toggle', base), ('version', alias.id)]
    finally:
        routing.is_subscription_owner = original_owner
    records['model_mutation_owner_gates'] = {'nonowner_subscription_writes': 0, 'owner_and_api_paths_work': True}


async def repro_subscription_task_state_isolation():
    inherited = {'chat_id': 'source', 'message_id': 'reply', 'task': 'title_generation',
                 'tool_ids': ['source-tool'], 'features': {'web_search': True}, 'user_message': {'files': ['source-file']}}
    for mid in ('subscription:claude:sonnet', 'subscription:codex:default', 'subscription:gemini:native', 'preset', 'api', 'arena-sub', 'arena-api'):
        model = {'id': mid, 'owned_by': 'openai'}
        if mid == 'preset':
            model['info'] = {'base_model_id': 'subscription:claude:sonnet'}
        models = {mid: model}
        native = mid not in ('api', 'arena-api')
        if mid.startswith('arena-'):
            chosen = 'subscription:claude:sonnet' if native else 'api'
            model.update(owned_by='arena', info={'meta': {'model_ids': [chosen]}})
            models[chosen] = {'id': chosen, 'owned_by': 'subscription' if native else 'openai'}
            routing.random = NS(choice=lambda candidates: candidates[0])
        request = Request({'type': 'http', 'method': 'POST', 'path': '/', 'headers': [],
            'app': NS(state=NS(MODELS=models)),
            'state': {'metadata': copy.deepcopy(inherited), 'terminal_id': 'source-terminal'}})
        original_state = request.state  # Deliberately populate cached Request._state.
        async def generate(task_request, form_data, user, **kwargs):
            if native:
                security.validate_chat_request_security(form_data, metadata=task_request.state.metadata, request=task_request)
                assert task_request is not request and task_request.state is not original_state
                assert task_request.state.metadata == {'chat_id': 'source', 'message_id': 'reply', 'task': 'title_generation'}
            else:
                assert task_request is request and form_data['metadata']['features']['web_search'] is True
            return {'ok': True}
        ns = {'is_subscription_model': routing.is_subscription_model, '_generate_task_completion': generate}
        wrapper = load_function('backend/open_webui/routers/tasks.py', 'generate_chat_completion', ns)
        await wrapper(request, {'model': mid, 'messages': [{'role': 'user', 'content': 'Write a title'}],
            'max_tokens': 1000, 'metadata': copy.deepcopy(inherited)}, NS(id='owner', role='admin'))
        ns.update(log=log, TASKS=NS(TITLE_GENERATION='title_generation'),
            Config=NS(get=AsyncMock(side_effect=lambda key: True if key == 'task.title.enable' else 'title template')),
            get_task_model_generation_config=AsyncMock(return_value=(mid, {'reasoning_effort': 'high', 'max_tokens': 50})),
            title_generation_template=AsyncMock(return_value='Write a title'),
            process_pipeline_inlet_filter=AsyncMock(side_effect=lambda request, payload, user, models: payload),
            apply_task_model_params=None)
        ns.update(collections=collections, JSONCodec=NS(loads=json.loads, JSONDecodeError=json.JSONDecodeError))
        load_function('backend/open_webui/utils/misc.py', 'deep_update', ns)
        load_function('backend/open_webui/utils/payload.py', 'apply_params_to_form_data', ns)
        load_function('backend/open_webui/routers/tasks.py', 'apply_task_model_params', ns)
        original_generate = ns['_generate_task_completion']
        async def title_result(task_request, form_data, user, **kwargs):
            assert form_data['reasoning_effort'] == 'high'
            return await original_generate(task_request, form_data, user, **kwargs)
        ns['_generate_task_completion'] = title_result
        title = load_function('backend/open_webui/routers/tasks.py', 'generate_title', ns, defaults=[])
        await title(request, {'model': mid, 'chat_id': 'source', 'messages': [{'role': 'user', 'content': 'hello'}]}, NS(id='owner', role='admin', email='synthetic@example.invalid'))
        assert request.state is original_state and original_state.metadata == inherited
        assert original_state.terminal_id == 'source-terminal'
    records['subscription_task_state_isolation'] = {'providers_and_alias': 4, 'cached_request_state_isolated': True, 'api_unchanged': True}


async def repro_mixed_fanout_defaults():
    route = node('backend/open_webui/main.py', 'chat_completion')
    body = next(item.body for item in route.body if isinstance(item, ast.Try))
    start = next(i for i, item in enumerate(body) if isinstance(item, ast.ImportFrom) and item.module == 'open_webui.subscriptions.controls')
    end = next(i for i in range(start, len(body)) if isinstance(body[i], ast.If) and ast.unparse(body[i].test) == 'model_info_params or request_params')
    loop = next(item for item in ast.walk(route) if isinstance(item, ast.For) and ast.unparse(item.target) == '(idx, entry)')
    first = next(i for i, item in enumerate(loop.body) if isinstance(item, ast.Assign) and ast.unparse(item.targets[0]) == 'model_form_data')
    last = next(i for i in range(first, len(loop.body)) if isinstance(loop.body[i], ast.Assign) and ast.unparse(loop.body[i].targets[0]) == "per_model_metadata['model']")
    def compile_stage(name, statements, ns):
        fn = ast.parse(f'async def {name}():\n    pass').body[0]
        fn.body = copy.deepcopy(statements) + ast.parse('return locals()').body
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'actual-fanout-defaults-stage', 'exec'), ns)
    defaults = {'temperature': .3, 'max_tokens': 1234}
    for primary_sub in (False, True):
        for provider in ('openai', 'ollama', 'subscription'):
            ns = {'copy': copy, 'Config': NS(get=AsyncMock(return_value=copy.deepcopy(defaults))),
                  'model_info': None, 'is_sub_model': primary_sub, 'missing_base_model': False,
                  'form_data': {'model': 'primary', 'params': {'reasoning_effort': 'high'}}, 'model_id': 'primary',
                  'HTTPException': HTTPException}
            load_function('backend/open_webui/utils/misc.py', 'merge_model_params', ns)
            compile_stage('initial', body[start:end+1], ns)
            ns.update(await ns['initial']())
            secondary = 'subscription:claude:sonnet' if provider == 'subscription' else provider
            ns.update(target_model_id=secondary, per_model_metadata={}, model={'id': 'primary'},
                request=NS(app=NS(state=NS(MODELS={secondary: {'id': secondary, 'owned_by': provider}}))),
                is_subscription_model=routing.is_subscription_model,
                Models=NS(get_model_by_id=AsyncMock(return_value=NS(params=NS(model_dump=lambda: {'temperature': .6})))) )
            compile_stage('target', loop.body[first:last], ns)
            payload = (await ns['target']())['model_form_data']
            if provider == 'subscription':
                controls.validate_subscription_chat_controls(payload)
                assert payload['params'] == {'reasoning_effort': 'high'}
            else:
                assert payload['params']['max_tokens'] == 1234
                assert payload['params']['temperature'] == (.6 if primary_sub else .3)
            assert ns['default_model_params'] == defaults
    records['mixed_fanout_defaults'] = {'primary_orders': 2, 'target_providers': 3, 'defaults_and_preset_precedence': True}


async def repro_preview_comments_and_bulk_lookup_boundaries():
    routing.is_subscription_owner = AsyncMock(side_effect=lambda user: user.id == 'owner')
    helpers = {}
    for name in ('_references_subscription_model', '_filter_subscription_model_records'):
        load_function('backend/open_webui/routers/models.py', name, helpers)
    router_module = types.ModuleType('open_webui.routers.models')
    router_module.__dict__.update(helpers)
    routers = types.ModuleType('open_webui.routers'); routers.__path__ = []
    sys.modules['open_webui.routers'] = routers
    sys.modules[router_module.__name__] = router_module
    sub = NS(id='subscription:claude:sonnet', name='Native', user_id='owner', base_model_id=None, is_active=True)
    preset = NS(id='private-preset', name='Synthetic Private Preset', user_id='owner', base_model_id=sub.id, is_active=True)
    api = NS(id='api', name='API', user_id='owner', base_model_id=None, is_active=True)
    rows = [sub, preset, api]
    request = NS(app=NS(state=NS(MODELS={r.id: {'id': r.id, 'base_model_id': r.base_model_id} for r in rows},
        BASE_MODELS=[{'id': sub.id, 'name': sub.name, 'owned_by': 'subscription'}])))
    ns = {'HTTPException': HTTPException, 'status': status,
          'Users': NS(get_user_by_id=AsyncMock(side_effect=lambda uid, **kw: NS(id=uid, name=uid))),
          'Models': NS(get_all_models=AsyncMock(return_value=rows)),
          'Groups': NS(get_groups_by_member_id=AsyncMock(return_value=[])),
          'AccessGrants': NS(get_accessible_resource_ids=AsyncMock(side_effect=lambda **kw: set(kw['resource_ids']))),
          'Knowledges': NS(get_knowledge_bases=AsyncMock(return_value=[])),
          'Tools': NS(get_tools=AsyncMock(return_value=[]))}
    preview = load_function('backend/open_webui/routers/users.py', 'get_user_preview', ns, defaults=[])
    for actor, target in (('other', 'owner'), ('owner', 'other'), ('owner', 'owner')):
        result = await preview(request, target, NS(id=actor, role='admin'), None)
        assert result['models']['total'] == (3 if actor == target == 'owner' else 1)
        assert (preset.id in str(result['models'])) == (actor == target == 'owner')
    group = NS(id='g', name='Group', permissions={})
    class Query:
        def where(self, *args): return self
        def with_for_update(self): return self
    ns.update(Group=NS(id=NS(in_=lambda ids: True)), select=lambda *args: Query(),
        Config=NS(get=AsyncMock(return_value={})), JSONCodec=NS(loads=json.loads, dumps=json.dumps),
        DEFAULT_USER_PERMISSIONS={}, fill_missing_permissions=lambda a, b: a,
        combine_permissions=lambda a, b: {**a, **b},
        resolve_group_default_models=lambda groups: ([api.id, sub.id, preset.id], None),
        group_default_models=lambda group: [api.id, sub.id, preset.id])
    ns['Groups'].get_group_by_id = AsyncMock(return_value=group)
    ns['Groups'].get_ancestor_ids = AsyncMock(return_value={'g'})
    group_preview = load_function('backend/open_webui/routers/groups.py', 'preview_group_access', ns, defaults=[])
    db = NS(execute=AsyncMock(return_value=NS(scalars=lambda: NS(all=lambda: []))))
    for actor in ('owner', 'other'):
        result = await group_preview(request, 'g', NS(id=actor, role='admin'), db)
        assert result['models']['total'] == 1
        assert result['default_models']['effective'] == [api.id]
        assert result['default_models']['local'] == [api.id]

    comments = {'HTTPException': HTTPException, 'SimpleNamespace': NS, 'select': lambda *args: Query(),
        'Config': NS(get=AsyncMock(return_value=True)), 'BYPASS_ADMIN_ACCESS_CONTROL': True,
        'resource_models': lambda: {'model': NS(id='synthetic-column')},
        'AccessGrants': NS(has_access=AsyncMock(return_value=False))}
    for name in ('_check_subscription_comment_access', 'provider_comment_resource', 'comment_resource', 'effective_comment_context'):
        load_function('backend/open_webui/utils/comments.py', name, comments)
    for actor in ('other', 'owner'):
        for row, mid in ((None, sub.id), (preset, preset.id), (api, api.id)):
            try:
                result = await comments['effective_comment_context']('model', mid, NS(id=actor, role='admin'),
                    NS(scalar=AsyncMock(return_value=row)), request=request)
                assert actor == 'owner' or mid == api.id
                assert result['resource_id'] == mid
            except HTTPException as exc:
                assert actor == 'other' and mid != api.id and exc.status_code == 404

    class CountingRegistry:
        def __init__(self, entries): self.entries = entries; self.gets = 0; self.reads = 0
        def get(self, key, default=None): self.gets += 1; return self.entries.get(key, default)
        def items(self): self.reads += 1; return self.entries.items()
    for count in (10, 100, 1000):
        public = [{'id': f'api-{i}', 'owned_by': 'openai'} for i in range(count)]
        registry = CountingRegistry({row['id']: row for row in public})
        req = NS(app=NS(state=NS(MODELS=registry)))
        visible = await helpers['_filter_subscription_model_records'](req, NS(id='other'), [*public, preset])
        assert len(visible) == count
        assert registry.reads == 1 and registry.gets == 0
    records['preview_comment_and_bulk_boundaries'] = {'owner_only_previews_and_comments': True,
        'group_models_not_shared': True, 'bulk_registry_snapshot_reads': 1, 'per_record_registry_io': 0}


async def repro_effective_fallback_before_persistence():
    sub, preset = 'subscription:claude:sonnet', 'legacy-api-preset'
    for bypass, actor, attachments in ((False, 'other', True), (True, 'other', True),
                                       (True, 'owner', True), (True, 'owner', False)):
        inserts = []
        async def check(user, model, **kwargs):
            if routing.is_subscription_model(model) and user.id != 'owner':
                raise HTTPException(403, 'Owner required')
        async def config(key, *defaults):
            return {'ui.default_models': sub, 'models.default_params': {}}.get(key, False)
        async def insert(*args, **kwargs):
            inserts.append(True)
            raise RuntimeError('FIRST_PERSISTENT_WRITE_REACHED')
        info = NS(id=preset, base_model_id='missing-api-base', params=NS(model_dump=lambda: {}))
        info.model_copy = lambda update: NS(**{**vars(info), **update})
        ns = {'copy': copy, 'asyncio': asyncio, 'HTTPException': HTTPException, 'status': status, 'log': log,
            'Models': NS(get_model_by_id=AsyncMock(return_value=info)), 'Config': NS(get=config),
            'BYPASS_MODEL_ACCESS_CONTROL': bypass, 'BYPASS_ADMIN_ACCESS_CONTROL': False,
            'ENABLE_CUSTOM_MODEL_FALLBACK': True, 'check_model_access': check,
            'merge_model_params': lambda a, b: {**a, **b}, 'is_saved_chat_id': lambda x: bool(x),
            'is_temporary_chat_id': lambda _: False, 'normalize_chat_variables': lambda x: x,
            'uuid4': uuid.uuid4, 'time': time, 'ChatForm': NS,
            'Chats': NS(get_chat_by_id=AsyncMock(return_value=None), insert_new_chat=insert)}
        route = load_function('backend/open_webui/main.py', 'chat_completion', ns, defaults=[None])
        registry = {preset: {'id': preset, 'owned_by': 'openai', 'info': {'base_model_id': 'missing-api-base'}},
                    sub: {'id': sub, 'owned_by': 'subscription'}}
        request = Request({'type': 'http', 'app': NS(state=NS(MODELS=registry, redis=None)), 'state': {}, 'headers': []})
        message = {'id': 'u1', 'role': 'user', 'content': 'hello'}
        if attachments: message['files'] = [{'type': 'file', 'id': 'synthetic-file'}]
        try:
            await route(request, {'model': preset, 'parent_id': None, 'id': 'a1',
                'messages': [{'role': 'user', 'content': 'hello'}], 'user_message': message},
                NS(id=actor, role='admin' if actor == 'owner' else 'user', name='Synthetic'))
            raise AssertionError('Expected rejection or synthetic persistence boundary')
        except HTTPException as exc:
            if actor != 'owner': assert exc.status_code == 403 and not inserts
            elif attachments: assert exc.status_code == 400 and not inserts
            else: assert exc.detail == 'FIRST_PERSISTENT_WRITE_REACHED' and len(inserts) == 1
    records['effective_fallback_prewrite'] = {'owner_and_text_gates_before_write': True, 'valid_owner_text_reaches_write': True}


async def repro_legacy_arena_selection_stays_fixed():
    previous = {key: getattr(routing, key, None) for key in ('random', 'resolve_arena_model', 'is_subscription_owner')}
    try:
        load_function('backend/open_webui/subscriptions/routing.py', 'resolve_arena_model', routing.__dict__)
        routing.HTTPException, routing.status = HTTPException, status
        routing.is_subscription_owner = AsyncMock(return_value=True)
        sub, preset = 'subscription:claude:sonnet', 'missing-base-preset'
        for fallback in (False, True):
            for selected in ('api', sub):
                draws, captured = [], []
                def draw(candidates):
                    assert selected in candidates
                    draws.append(selected)
                    return selected
                routing.random = NS(choice=draw)
                registry = {preset: {'id': preset, 'info': {'base_model_id': 'missing'}},
                    'arena': {'id': 'arena', 'owned_by': 'arena', 'info': {'meta': {'model_ids': ['api', sub]}}},
                    'api': {'id': 'api', 'owned_by': 'openai'}, sub: {'id': sub, 'owned_by': 'subscription'}}
                info = NS(id=preset, base_model_id='missing', params=NS(model_dump=lambda: {}))
                info.model_copy = lambda update: NS(**{**vars(info), **update})
                async def model_info(mid): return info if mid == preset else None
                async def config(key, *defaults):
                    return {'ui.default_models': 'arena', 'models.default_params': {'max_tokens': 1000}}.get(key, False)
                async def payload(request, body, user, metadata, model):
                    captured.append((copy.deepcopy(body), metadata['selected_model_id'], model['id']))
                    raise RuntimeError('SYNTHETIC_LEGACY_DISPATCH')
                ns = {'copy': copy, 'asyncio': asyncio, 'HTTPException': HTTPException, 'status': status, 'log': log, 'time': time,
                    'Models': NS(get_model_by_id=model_info), 'Config': NS(get=config), 'BYPASS_MODEL_ACCESS_CONTROL': False,
                    'BYPASS_ADMIN_ACCESS_CONTROL': False, 'ENABLE_CUSTOM_MODEL_FALLBACK': True, 'check_model_access': AsyncMock(),
                    'merge_model_params': lambda a, b: {**a, **b}, 'apply_model_controls': lambda params, *_: params,
                    'is_saved_chat_id': lambda _: False, 'is_temporary_chat_id': lambda _: False,
                    'normalize_chat_variables': lambda x: x, 'uuid4': uuid.uuid4, 'process_chat_payload': payload,
                    'Chats': NS(get_chat_by_id=AsyncMock(return_value=None))}
                route = load_function('backend/open_webui/main.py', 'chat_completion', ns, defaults=[None])
                request = Request({'type': 'http', 'app': NS(state=NS(MODELS=registry, redis=None)), 'state': {}, 'headers': []})
                try:
                    await route(request, {'model': preset if fallback else 'arena', 'params': {'reasoning_effort': 'high'},
                        'messages': [{'role': 'user', 'content': 'hello'}]}, NS(id='owner', role='admin', name='Synthetic'))
                except HTTPException as exc:
                    assert exc.detail == 'SYNTHETIC_LEGACY_DISPATCH', exc.detail
                assert draws == [selected] and len(captured) == 1
                body, meta_model, actual_model = captured[0]
                assert body['model'] == meta_model == actual_model == selected
                assert body['params']['reasoning_effort'] == 'high'
                assert ('max_tokens' in body['params']) == (selected == 'api')
    finally:
        for key, value in previous.items(): setattr(routing, key, value)
    records['legacy_arena_selection'] = {'single_draw': True, 'fallback_and_primary': True, 'native_and_api_defaults': True}


async def repro_auxiliary_model_metadata_privacy():
    import base64, io
    from contextlib import asynccontextmanager
    from fastapi.responses import Response, RedirectResponse, StreamingResponse
    from sqlalchemy import JSON, Column, String, create_engine, select, or_
    from sqlalchemy.orm import declarative_base
    base = declarative_base()
    class Row(base):
        __tablename__ = 'synthetic_model'
        id = Column(String, primary_key=True); base_model_id = Column(String); meta = Column(JSON)
    engine = create_engine('sqlite:///:memory:')
    base.metadata.create_all(engine)
    sub = 'subscription:claude:sonnet'
    rows = [{'id': mid, 'base_model_id': parent, 'meta': {'tags': [{'name': mid}]}}
        for mid, parent in ((sub, None), ('private-preset', sub), ('cached-native', 'api'), ('api', None), ('api-preset', 'api'))]
    with engine.begin() as conn: conn.execute(Row.__table__.insert(), rows)
    class Database:
        async def execute(self, stmt):
            with engine.connect() as conn: values = list(conn.execute(stmt).scalars())
            return NS(scalars=lambda: NS(all=lambda: values))
    @asynccontextmanager
    async def context(db): yield db
    tag_method = load_function('backend/open_webui/models/models.py', 'get_all_tags',
        {'get_async_db_context': context, 'select': select, 'Model': Row, 'or_': or_})
    async def tags(**kwargs): return await tag_method(NS(), **kwargs)
    registry = {'cached-native': {'id': 'cached-native', 'info': {'base_model_id': sub}}}
    request = Request({'type': 'http', 'app': NS(state=NS(MODELS=registry)), 'state': {}, 'headers': []})
    ns = {'Models': NS(get_all_tags=tags), 'BYPASS_ADMIN_ACCESS_CONTROL': True}
    for name in ('_references_subscription_model', '_subscription_model_tag_filter', 'get_base_model_tags', 'get_model_tags'):
        load_function('backend/open_webui/routers/models.py', name, ns, defaults=[] if name.startswith('get_') else None)
    previous_owner = routing.is_subscription_owner
    try:
        for actor in ('owner', 'other'):
            routing.is_subscription_owner = AsyncMock(return_value=actor == 'owner')
            user = NS(id=actor, role='admin')
            actual = set(await ns['get_base_model_tags'](request, user, Database())) | set(await ns['get_model_tags'](request, user, Database()))
            assert actual == ({r['id'] for r in rows} if actor == 'owner' else {'api', 'api-preset'})
            private_bytes = b'synthetic-avatar'
            meta = {'profile_image_url': 'data:image/png;base64,' + base64.b64encode(private_bytes).decode()}
            images = {**ns, 'Models': NS(get_model_by_id=AsyncMock(), get_model_meta_by_id=AsyncMock(return_value=(meta, 'owner', 1))),
                'BYPASS_MODEL_ACCESS_CONTROL': True, 'Config': NS(get=AsyncMock(return_value=[])), 'ENABLE_PROFILE_IMAGE_URL_FORWARDING': False,
                'PROFILE_IMAGE_ALLOWED_MIME_TYPES': {'image/png'}, 'Response': Response, 'RedirectResponse': RedirectResponse,
                'StreamingResponse': StreamingResponse, 'base64': base64, 'io': io, 'status': status}
            image = load_function('backend/open_webui/routers/models.py', 'get_model_profile_image', images, defaults=[])
            for mid, parent in ((sub, None), ('private-preset', sub), ('cached-native', 'api'), ('api', None)):
                images['Models'].get_model_by_id.return_value = NS(id=mid, base_model_id=parent)
                response = await image(request, mid, user, None)
                if actor == 'other' and mid != 'api':
                    assert response.status_code == 302 and response.headers['location'] == '/static/favicon.png'
                else:
                    assert response.status_code == 200
                    assert b''.join([chunk async for chunk in response.body_iterator]) == private_bytes
    finally:
        routing.is_subscription_owner = previous_owner
        engine.dispose()
    records['auxiliary_model_metadata'] = {'owner_tags_and_images': True, 'nonowner_native_metadata_hidden': True, 'api_preserved': True}


async def main():
    await repro_direct(); await repro_saved(); await repro_native_reasoning_menu(); await repro_post_filter_security_guard(); await repro_cache(); await repro_stream_scope(); await repro_arena(); await repro_failed_event(); await repro_bulk_model_owner_guards()
    await repro_compaction_subscription_defaults(); await repro_custom_subscription_menus()
    await repro_subscription_log_privacy(); await repro_saved_history_before_media_io(); await repro_model_mutation_owner_gates()
    await repro_subscription_task_state_isolation(); await repro_mixed_fanout_defaults()
    await repro_preview_comments_and_bulk_lookup_boundaries(); await repro_effective_fallback_before_persistence()
    await repro_legacy_arena_selection_stays_fixed(); await repro_auxiliary_model_metadata_privacy()
    print(json.dumps(records,indent=2))

if __name__ == '__main__':
    asyncio.run(main())

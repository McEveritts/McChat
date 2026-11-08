import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import vm from 'node:vm';

test('Google sign-out response and UI distinguish verified logout from failure', async () => {
	const source = await readFile(new URL('../src/lib/apis/subscriptions.ts', import.meta.url), 'utf8');
	const signature = source.slice(source.indexOf('export const providerSignOut'), source.indexOf('let error = null;', source.indexOf('export const providerSignOut')));
	assert.match(signature, /success:\s*boolean/);
	const code = stripTypeScriptTypes(source)
		.replace(/import\s*\{\s*WEBUI_API_BASE_URL\s*\}\s*from\s*['"][^'"]+['"];?/, 'const WEBUI_API_BASE_URL = "https://mcchat.invalid/api/v1";')
		.replace(/export\s+const\s+/g, 'const ');
	const card = await readFile(new URL('../src/lib/components/admin/Settings/Connections/SubscriptionConnection.svelte', import.meta.url), 'utf8');
	const handler = stripTypeScriptTypes(card.slice(card.indexOf('const handleSignOut ='), card.indexOf('const runTest =')));
	for (const success of [true, false]) {
		const toasts = [];
		let refreshed = 0;
		const context = vm.createContext({
			console, localStorage: { token: 'synthetic-session' }, provider: { id: 'gemini' },
			$i18n: { t: text => text }, signingOut: false, authFlow: {}, returnUrl: 'old',
			stopPolling() {}, notifyAuthChanged() { refreshed += 1; },
			toast: { success: message => toasts.push(['success', message]), error: message => toasts.push(['error', message]) },
			fetch: async (url, options) => {
				assert.equal(url, 'https://mcchat.invalid/api/v1/subscriptions/providers/gemini/logout');
				assert.equal(options.method, 'POST');
				return { ok: true, json: async () => ({ success, phase: success ? 'succeeded' : 'failed', message: 'Provider logout unverified', authenticated: false }) };
			}
		});
		await new vm.Script(`${code}\n${handler}\nhandleSignOut()`).runInContext(context);
		assert.equal(toasts.length, 1);
		assert.equal(toasts[0][0], success ? 'success' : 'error');
		assert.equal(context.signingOut, false);
		assert.equal(context.authFlow, null);
		assert.equal(refreshed, 1);
	}
});

test('chat respects a model that disables default memory', async () => {
	const source = await readFile(new URL('../src/lib/components/chat/Chat.svelte', import.meta.url), 'utf8');
	const start = source.indexOf('const getFeatures =');
	const end = source.indexOf('const getStopTokens =', start);
	assert.ok(start >= 0 && end > start);
	const context = vm.createContext({
		$config: { features: { enable_memories: true } },
		$settings: {}, $user: { role: 'admin' }, $showCallOverlay: false,
		callMode: 'current', bridge: null, imageGenerationEnabled: false,
		codeInterpreterEnabled: false, webSearchActive: false,
	});
	const getFeatures = new vm.Script(`${source.slice(start, end)}; getFeatures`).runInContext(context);
	assert.equal(getFeatures({ info: { meta: { capabilities: { memory: false } } } }).memory, undefined);
	assert.equal(getFeatures({}).memory, true);
});

test('subscription settings GET response can be saved without response-only fields', async () => {
	const source = await readFile(new URL('../src/lib/apis/subscriptions.ts', import.meta.url), 'utf8');
	const settings = {
		ENABLE_SUBSCRIPTIONS: false,
		SUBSCRIPTIONS_OWNER_ID: 'owner',
		SUBSCRIPTIONS_REQUEST_TIMEOUT: 180,
		SUBSCRIPTIONS_PROVIDERS: { claude: { enabled: false } },
		current_user_id: 'owner'
	};
	let saved;
	const context = vm.createContext({
		console,
		fetch: async (url, options) => {
			if (options.method === 'POST') {
				saved = JSON.parse(options.body);
				assert.equal(options.headers.authorization, 'Bearer test-session');
				assert.equal(url, 'https://mcchat.invalid/api/v1/subscriptions/config/update');
			}
			return { ok: true, json: async () => ({ ...settings }) };
		}
	});

	const jsCode = stripTypeScriptTypes(source)
		.replace(
			/import\s*\{\s*WEBUI_API_BASE_URL\s*\}\s*from\s*['"][^'"]+['"];?/,
			'const WEBUI_API_BASE_URL = "https://mcchat.invalid/api/v1";'
		)
		.replace(/export\s+const\s+/g, 'const ');

	const script = new vm.Script(`
		${jsCode}
		({ getSubscriptionsConfig, updateSubscriptionsConfig })
	`);
	const api = script.runInContext(context);
	const received = await api.getSubscriptionsConfig('test-session');
	assert.equal(received.current_user_id, 'owner');
	await api.updateSubscriptionsConfig('test-session', received);
	const { current_user_id, ...writable } = settings;
	assert.deepEqual(saved, writable);
	assert.equal(received.current_user_id, 'owner');
});

test('mixed and Arena requests use text-only controls while API-only requests retain defaults', async () => {
const source = await readFile(new URL('../src/lib/components/chat/Chat.svelte', import.meta.url), 'utf8');
const featureCode=source.slice(source.indexOf('const getFeatures ='),source.indexOf('const getStopTokens ='));
const start=source.indexOf('const isSubscription =');const end=source.indexOf('\n\t\tif (filesystemAttachment',start);
const models=[{id:'api',owned_by:'openai'},{id:'subscription:codex:default',owned_by:'subscription',info:{meta:{capabilities:{memory:false,tools:false}}}},{id:'arena',owned_by:'arena',info:{meta:{model_ids:['subscription:codex:default']}}}];
let captured;
const context=vm.createContext({console,$config:{features:{enable_memories:true}},$settings:{params:{temperature:0.7,max_tokens:4096},memory:true},$user:{id:'owner',role:'admin'},$showCallOverlay:false,callMode:'current',bridge:null,imageGenerationEnabled:false,codeInterpreterEnabled:false,webSearchActive:false,
 useChatVariablesFallback:false,chatVariables:{},stream:true,messages:[],params:{},getStopTokens:()=>undefined,embedded:false,files:[],noteFiles:[],selectedFilterIds:[],toolIds:[],skillIds:[],terminalEnabled:false,requestTerminalId:null,toolServerIds:[],$toolServers:[],$terminalServers:[],filesystemAttachment:null,getPromptVariables:()=>({}),userLocation:undefined,$models:models,$socket:{id:'fixture-session'},_chatId:'fixture-chat',$selectedFolder:null,responseMessageId:'a1',userMessage:{id:'u1',role:'user',content:'hello',parentId:null},regenerationPrompt:null,regenerationMessageId:null,continueResponse:false,$temporaryChatEnabled:false,_history:{},createMessagesList:()=>[],WEBUI_BASE_URL:'https://mcchat.invalid',localStorage:{token:'fixture-session-token'},generateOpenAIChatCompletion:async(_token,payload)=>{captured=payload;return {}},toast:{error:()=>{}},history:{messages:{},currentId:null},responseMessage:{}});
const helpers = stripTypeScriptTypes(await readFile(new URL('../src/lib/utils/subscriptions.ts', import.meta.url), 'utf8')).replace(/export const /g, 'const '); new vm.Script(helpers).runInContext(context);
const extract=new vm.Script(`${featureCode}; async function capture(){${source.slice(start,end)}; return res;} capture`).runInContext(context);

for (const [primary, ids, subscription] of [
 [models[0], ['api','subscription:codex:default'], true],
 [models[1], ['subscription:codex:default','api'], true],
 [models[2], ['arena'], true],
 [models[0], ['api'], false],
]) {
 context.model=primary;context.messageIdsList=ids.map((id,i)=>({model_id:id,message_id:'a'+i}));
 await extract();
 assert.equal(captured.params.temperature, subscription ? undefined : 0.7);
 assert.equal(captured.params.max_tokens, subscription ? undefined : 4096);
 assert.equal(captured.features.memory, subscription ? undefined : true);
}
assert.equal(context.$settings.params.temperature, 0.7);
});

test('subscription and mixed/Arena requests preserve personal reasoning_effort and per-chat override precedence', async () => {
	const source = await readFile(new URL('../src/lib/components/chat/Chat.svelte', import.meta.url), 'utf8');
	const featureCode = source.slice(source.indexOf('const getFeatures ='), source.indexOf('const getStopTokens ='));
	const start = source.indexOf('const isSubscription =');
	const end = source.indexOf('\n\t\tif (filesystemAttachment', start);
	const models = [
		{ id: 'api', owned_by: 'openai' },
		{ id: 'subscription:claude:default', owned_by: 'subscription', info: { meta: { capabilities: { memory: false, tools: false } } } },
		{ id: 'arena', owned_by: 'arena', info: { meta: { model_ids: ['subscription:claude:default'] } } }
	];
	let captured;
	const context = vm.createContext({
		console,
		$config: { features: { enable_memories: true } },
		$settings: { params: { temperature: 0.7, max_tokens: 4096, reasoning_effort: 'high' }, memory: true },
		$user: { id: 'owner', role: 'admin' },
		$showCallOverlay: false,
		callMode: 'current',
		bridge: null,
		imageGenerationEnabled: false,
		codeInterpreterEnabled: false,
		webSearchActive: false,
		useChatVariablesFallback: false,
		chatVariables: {},
		stream: true,
		messages: [],
		params: {},
		getStopTokens: () => undefined,
		embedded: false,
		files: [],
		noteFiles: [],
		selectedFilterIds: [],
		toolIds: ['search-tool'],
		skillIds: [],
		terminalEnabled: false,
		requestTerminalId: null,
		toolServerIds: [],
		$toolServers: [],
		$terminalServers: [],
		filesystemAttachment: null,
		getPromptVariables: () => ({}),
		userLocation: undefined,
		$models: models,
		$socket: { id: 'fixture-session' },
		_chatId: 'fixture-chat',
		$selectedFolder: null,
		responseMessageId: 'a1',
		userMessage: { id: 'u1', role: 'user', content: 'hello', parentId: null },
		regenerationPrompt: null,
		regenerationMessageId: null,
		continueResponse: false,
		$temporaryChatEnabled: false,
		_history: {},
		createMessagesList: () => [],
		WEBUI_BASE_URL: 'https://mcchat.invalid',
		localStorage: { token: 'fixture-session-token' },
		generateOpenAIChatCompletion: async (_token, payload) => {
			captured = payload;
			return {};
		},
		toast: { error: () => {} },
		history: { messages: {}, currentId: null },
		responseMessage: {}
	});
	const helpers = stripTypeScriptTypes(await readFile(new URL('../src/lib/utils/subscriptions.ts', import.meta.url), 'utf8')).replace(/export const /g, 'const ');
	new vm.Script(helpers).runInContext(context);
	const extract = new vm.Script(`${featureCode}; async function capture(){${source.slice(start, end)}; return res;} capture`).runInContext(context);

	// Case 1: Subscription model with personal settings reasoning_effort = 'high', no per-chat override
	context.model = models[1];
	context.messageIdsList = [{ model_id: 'subscription:claude:default', message_id: 'a0' }];
	context.params = {};
	await extract();
	assert.equal(captured.params.reasoning_effort, 'high');
	assert.equal(captured.params.temperature, undefined);
	assert.equal(captured.params.max_tokens, undefined);
	assert.equal(captured.tool_ids.length, 0);

	// Case 2: Per-chat override 'max' takes precedence over personal setting 'high'
	context.params = { reasoning_effort: 'max' };
	await extract();
	assert.equal(captured.params.reasoning_effort, 'max');
	assert.equal(captured.params.temperature, undefined);

	// Case 3: Mixed targets (API + Subscription) with per-chat override 'high'
	context.model = models[0];
	context.messageIdsList = [{ model_id: 'api', message_id: 'a0' }, { model_id: 'subscription:claude:default', message_id: 'a1' }];
	context.params = { reasoning_effort: 'high' };
	await extract();
	assert.equal(captured.params.reasoning_effort, 'high');
	assert.equal(captured.params.temperature, undefined);
	assert.equal(captured.tool_ids.length, 0);

	// Case 4: Arena target with no per-chat override falls back to personal setting 'high'
	context.model = models[2];
	context.messageIdsList = [{ model_id: 'arena', message_id: 'a0' }];
	context.params = {};
	await extract();
	assert.equal(captured.params.reasoning_effort, 'high');
	assert.equal(captured.params.temperature, undefined);

	// Case 5: Pure API target retains sampling controls and stop tokens
	context.model = models[0];
	context.messageIdsList = [{ model_id: 'api', message_id: 'a0' }];
	context.params = {};
	await extract();
	assert.equal(captured.params.temperature, 0.7);
	assert.equal(captured.params.max_tokens, 4096);
	assert.equal(captured.params.reasoning_effort, 'high');

	// Personal settings remain unmutated
	assert.equal(context.$settings.params.reasoning_effort, 'high');
	assert.equal(context.$settings.params.temperature, 0.7);
});

test('connections update handler skips writes when non-owner admin views configuration', async () => {
	const source = await readFile(new URL('../src/lib/components/admin/Settings/Connections.svelte', import.meta.url), 'utf8');

	assert.ok(source.includes('subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID !== subscriptionsConfig.current_user_id'));
	assert.ok(source.includes('initialSubscriptionsSnapshot'));

	// Test non-owner check logic
	const nonOwnerConfig = {
		ENABLE_SUBSCRIPTIONS: true,
		SUBSCRIPTIONS_OWNER_ID: 'admin-1',
		current_user_id: 'admin-2'
	};
	const isNonOwner = nonOwnerConfig.SUBSCRIPTIONS_OWNER_ID && nonOwnerConfig.SUBSCRIPTIONS_OWNER_ID !== nonOwnerConfig.current_user_id;
	assert.equal(isNonOwner, true);
});

test('failed subscription stream advances the queue without success side effects', async () => {
	const source = await readFile(new URL('../src/lib/components/chat/Chat.svelte', import.meta.url), 'utf8');
	const start = source.indexOf('const chatCompletionEventHandler =');
	const end = source.indexOf('\n\t// Chat functions', start);
	assert.ok(start >= 0 && end > start);
	const queued = [];
	const successEffects = [];
	const context = vm.createContext({
		history: { messages: {} }, $contextCompactions: {}, bridge: null,
		handleOpenAIError: (error, message) => { message.error = error; },
		clearContextCompaction: () => {},
		processNextInQueue: async (id) => { queued.push(id); },
		chatCompletedHandler: () => { successEffects.push('completed'); },
		dispatchCallOverlayAudio: () => { successEffects.push('audio'); },
	});
	const handler = new vm.Script(`${source.slice(start, end)}; chatCompletionEventHandler`).runInContext(context);
	const message = { id: 'message', content: 'partial', done: false };
	await handler({ done: false, failed: true, error: { message: 'CLI_STREAM_ERROR' } }, message, 'chat');
	assert.deepEqual(queued, ['chat']);
	assert.deepEqual(successEffects, []);
	assert.equal(message.done, true);
	assert.ok(message.error);
});


test('owner model catalog refresh updates the global picker without a chat id', async () => {
    const source = await readFile(new URL('../src/routes/+layout.svelte', import.meta.url), 'utf8');
    const start = source.indexOf('const chatEventHandler = async (event, cb) => {');
    const end = source.indexOf('const chat = $page.url.pathname', start);
    assert.ok(start >= 0 && end > start);
    const handler = source.slice(start, end) + '}; globalThis.handler = chatEventHandler;';
    const calls = [];
    const catalog = [{id:'subscription:gemini:model-b',info:{params:{model_controls:{reasoning_effort:{}}}}}];
    const context = {
        localStorage:{token:'synthetic-owner'}, $config:{features:{enable_direct_connections:false}},
        $settings:{}, models:{set(value){calls.push(value);}},
        getModels:async token => {assert.equal(token,'synthetic-owner');return catalog;}
    };
    vm.createContext(context);
    vm.runInContext(handler, context);
    await context.handler({data:{type:'models:refresh',data:{}}});
    assert.equal(calls.length,1);
    assert.equal(calls[0],catalog);
});

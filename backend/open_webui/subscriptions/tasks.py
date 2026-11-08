"""Isolate server-generated text tasks from their source chat's tool context."""
from __future__ import annotations

import copy

from starlette.datastructures import State


def prepare_subscription_task_request(request, metadata: dict):
    """Keep trace identifiers when the server constructs a separate text task.

    Ordinary chat requests retain their explicit controls and strict validation.
    """
    trace_keys = ('chat_id', 'message_id', 'user_message_id', 'session_id', 'task_id', 'task', 'id')
    task_metadata = {key: metadata[key] for key in trace_keys
        if key in metadata and (metadata[key] is None or isinstance(metadata[key], str))}
    state = {'metadata': task_metadata, 'internal': getattr(request.state, 'internal', False) is True}
    task_request = copy.copy(request)
    if hasattr(request, 'scope'):
        task_request.scope = {**request.scope, 'state': state}
        task_request._state = State(state)
    else:
        task_request.state = State(state)
    return task_request, task_metadata

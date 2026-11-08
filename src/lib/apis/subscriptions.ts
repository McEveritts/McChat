import { WEBUI_API_BASE_URL } from '$lib/constants';

export const getErrorMessage = (err: any, fallback = 'Server connection failed') => {
	if (Array.isArray(err)) return fallback;
	return (
		[
			err?.detail?.error?.message,
			err?.detail?.message,
			err?.detail?.content,
			err?.detail?.error,
			err?.detail,
			err?.error?.message,
			err?.error?.content,
			err?.error,
			err?.message,
			err?.content,
			err
		].find((value) => typeof value === 'string' && value.trim()) ?? fallback
	);
};

export type SubscriptionProviderStatus = {
	id: string;
	name: string;
	enabled: boolean;
	installed: boolean;
	binary_path: string;
	config_dir: string;
	authenticated: boolean;
	execution_available?: boolean;
	reason?: string;
	status: 'authenticated' | 'needs_login' | 'not_installed' | 'error' | string;
	version: string | null;
	message: string;
	models: { id: string; name: string }[];
};

export type SubscriptionsConfig = {
	ENABLE_SUBSCRIPTIONS: boolean;
	SUBSCRIPTIONS_OWNER_ID: string;
	SUBSCRIPTIONS_REQUEST_TIMEOUT: number;
	SUBSCRIPTIONS_PROVIDERS: Record<string, any>;
	current_user_id?: string;
};

export const getSubscriptionsConfig = async (token: string = ''): Promise<SubscriptionsConfig | null> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/subscriptions/config`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = getErrorMessage(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const updateSubscriptionsConfig = async (
	token: string = '',
	config: SubscriptionsConfig
): Promise<SubscriptionsConfig | null> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/subscriptions/config/update`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		},
		body: JSON.stringify({
			ENABLE_SUBSCRIPTIONS: config.ENABLE_SUBSCRIPTIONS,
			SUBSCRIPTIONS_OWNER_ID: config.SUBSCRIPTIONS_OWNER_ID,
			SUBSCRIPTIONS_REQUEST_TIMEOUT: config.SUBSCRIPTIONS_REQUEST_TIMEOUT,
			SUBSCRIPTIONS_PROVIDERS: config.SUBSCRIPTIONS_PROVIDERS
		})
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = getErrorMessage(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getSubscriptionProviders = async (
	token: string = ''
): Promise<SubscriptionProviderStatus[] | null> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/subscriptions/providers`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = getErrorMessage(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const testSubscriptionProvider = async (
	token: string = '',
	providerId: string,
	prompt: string = 'Hello, reply with "pong" and nothing else.',
	model?: string
): Promise<{ success: boolean; elapsed_ms: number; message: string; response?: string } | null> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/subscriptions/providers/${providerId}/test`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		},
		body: JSON.stringify({ prompt, model })
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = getErrorMessage(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export type ProviderAuthFlowState = {
	success?: boolean;
	phase: 'idle' | 'starting' | 'waiting' | 'verifying' | 'succeeded' | 'failed' | 'cancelled';
	flow_id: string | null;
	authorization_url: string | null;
	message: string | null;
	authenticated: boolean;
	models?: { id: string; name: string }[];
	version?: string | null;
	epoch?: string | number | null;
	profile_identity?: string | null;
};

export const startProviderSignIn = async (
	token: string = '',
	providerId: string
): Promise<ProviderAuthFlowState | null> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/subscriptions/providers/${providerId}/signin`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = getErrorMessage(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getProviderAuthStatus = async (
	token: string = '',
	providerId: string
): Promise<ProviderAuthFlowState | null> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/subscriptions/providers/${providerId}/auth`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = getErrorMessage(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const completeProviderSignIn = async (
	token: string = '',
	providerId: string,
	flowId: string,
	callbackUrl: string
): Promise<ProviderAuthFlowState | null> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/subscriptions/providers/${providerId}/callback`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		},
		body: JSON.stringify({ flow_id: flowId, callback_url: callbackUrl })
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = getErrorMessage(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const cancelProviderSignIn = async (
	token: string = '',
	providerId: string,
	flowId: string
): Promise<ProviderAuthFlowState | null> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/subscriptions/providers/${providerId}/cancel`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		},
		body: JSON.stringify({ flow_id: flowId })
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = getErrorMessage(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const providerSignOut = async (
	token: string = '',
	providerId: string
): Promise<{ success: boolean; phase: string; message: string; authenticated: boolean } | null> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/subscriptions/providers/${providerId}/logout`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			...(token && { authorization: `Bearer ${token}` })
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = getErrorMessage(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

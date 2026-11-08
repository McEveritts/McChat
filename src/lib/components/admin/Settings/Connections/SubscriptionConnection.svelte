<script lang="ts">
	import { getContext, onDestroy } from 'svelte';
	import { toast } from 'svelte-sonner';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Switch from '$lib/components/common/Switch.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import Check from '$lib/components/icons/Check.svelte';
	import {
		testSubscriptionProvider,
		startProviderSignIn,
		getProviderAuthStatus,
		completeProviderSignIn,
		cancelProviderSignIn,
		providerSignOut,
		type SubscriptionProviderStatus,
		type ProviderAuthFlowState
	} from '$lib/apis/subscriptions';

	const i18n: any = getContext('i18n');

	export let provider: SubscriptionProviderStatus;
	export let config: any = {};
	export let onUpdate: () => void = () => {};
	export let onAuthChanged: () => void = () => {};
	export let disabled = false;

	const notifyAuthChanged = () => {
		if (onAuthChanged) {
			onAuthChanged();
		} else {
			onUpdate();
		}
	};

	let testing = false;
	let testResult: { success: boolean; elapsed_ms: number; message: string; response?: string } | null = null;
	let showAdvanced = false;

	let authFlow: ProviderAuthFlowState | null = null;
	let startingAuth = false;
	let completingAuth = false;
	let signingOut = false;
	let returnUrl = '';
	let pollTimer: any = null;

	const stopPolling = () => {
		if (pollTimer) {
			clearInterval(pollTimer);
			pollTimer = null;
		}
	};

	onDestroy(() => {
		stopPolling();
	});

	const startPolling = () => {
		stopPolling();
		pollTimer = setInterval(async () => {
			if (!authFlow || authFlow.phase === 'succeeded' || authFlow.phase === 'failed' || authFlow.phase === 'cancelled') {
				stopPolling();
				return;
			}
			try {
				const res = await getProviderAuthStatus(localStorage.token, provider.id);
				if (res) {
					authFlow = res;
					if (res.phase === 'succeeded') {
						stopPolling();
						toast.success($i18n.t('Successfully signed in with Google!'));
						notifyAuthChanged();
					} else if (res.phase === 'failed') {
						stopPolling();
						toast.error(res.message || $i18n.t('Google sign-in failed.'));
					}
				}
			} catch (e) {
				// ignore poll transient errors
			}
		}, 2000);
	};

	const handleStartSignIn = async () => {
		startingAuth = true;
		testResult = null;
		try {
			const res = await startProviderSignIn(localStorage.token, provider.id);
			if (res) {
				authFlow = res;
				returnUrl = '';
				startPolling();
			}
		} catch (err: any) {
			toast.error(`${err}`);
		} finally {
			startingAuth = false;
		}
	};

	const handleCompleteSignIn = async () => {
		if (!authFlow || !authFlow.flow_id || !returnUrl.trim()) return;
		completingAuth = true;
		try {
			const res = await completeProviderSignIn(localStorage.token, provider.id, authFlow.flow_id, returnUrl.trim());
			if (res) {
				authFlow = res;
				if (res.phase === 'succeeded') {
					stopPolling();
					toast.success($i18n.t('Successfully signed in with Google!'));
					notifyAuthChanged();
				}
			}
		} catch (err: any) {
			toast.error(`${err}`);
		} finally {
			completingAuth = false;
		}
	};

	const handleCancelSignIn = async () => {
		stopPolling();
		if (authFlow && authFlow.flow_id) {
			try {
				await cancelProviderSignIn(localStorage.token, provider.id, authFlow.flow_id);
			} catch (e) {
				// ignore cancel errors
			}
		}
		authFlow = null;
		returnUrl = '';
	};

	const handleSignOut = async () => {
		signingOut = true;
		stopPolling();
		try {
			const res = await providerSignOut(localStorage.token, provider.id);
			authFlow = null;
			returnUrl = '';
			if (res?.success === true) {
				toast.success($i18n.t('Signed out from Google Antigravity.'));
			} else {
				toast.error(res?.message || $i18n.t('Sign-out could not be verified. Please try again.'));
			}
		} catch (err: any) {
			toast.error(`${err}`);
		} finally {
			signingOut = false;
			notifyAuthChanged();
		}
	};

	const runTest = async () => {
		if (provider.execution_available === false) {
			toast.error(provider.reason || $i18n.t('Provider execution is unavailable.'));
			return;
		}

		testing = true;
		testResult = null;
		try {
			const res = await testSubscriptionProvider(localStorage.token, provider.id);
			if (res) {
				testResult = res;
				if (res.success) {
					toast.success($i18n.t(`Connected to ${provider.name} in ${res.elapsed_ms}ms`));
					notifyAuthChanged();
				} else {
					toast.error(res.message);
				}
			}
		} catch (err: any) {
			toast.error(`${err}`);
			testResult = {
				success: false,
				elapsed_ms: 0,
				message: `${err}`
			};
		} finally {
			testing = false;
		}
	};
</script>

<div class="flex flex-col gap-2 rounded-xl border border-gray-100 bg-white/50 p-3.5 dark:border-gray-800/80 dark:bg-gray-900/30">
	<div class="flex items-center justify-between gap-3">
		<div class="flex items-center gap-2.5">
			<div class="flex size-7 shrink-0 items-center justify-center rounded-lg bg-gray-100 text-xs font-semibold text-gray-700 dark:bg-gray-800 dark:text-gray-300">
				{#if provider.id === 'claude'}
					C
				{:else if provider.id === 'codex'}
					O
				{:else if provider.id === 'gemini'}
					G
				{:else}
					S
				{/if}
			</div>

			<div>
				<div class="flex items-center gap-2">
					<span class="font-medium text-gray-900 dark:text-gray-100 text-xs sm:text-sm">
						{provider.name}
					</span>

					{#if provider.execution_available === false}
						<span class="inline-flex items-center gap-1 rounded-full bg-red-500/10 px-2 py-0.5 text-xs font-medium text-red-600 dark:text-red-400">
							<span class="size-1 rounded-full bg-red-500"></span>
							{$i18n.t('Unavailable')}
						</span>
					{:else if provider.authenticated}
						<span class="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-600 dark:text-emerald-400">
							<span class="size-1 rounded-full bg-emerald-500"></span>
							{$i18n.t('Ready')}
						</span>
					{:else if provider.installed && provider.status === 'unknown'}
						<span class="inline-flex items-center gap-1 rounded-full bg-amber-500/10 px-2 py-0.5 text-xs font-medium text-amber-600 dark:text-amber-400">
							{$i18n.t('Account unverified')}
						</span>
					{:else if provider.installed}
						<span class="inline-flex items-center gap-1 rounded-full bg-amber-500/10 px-2 py-0.5 text-xs font-medium text-amber-600 dark:text-amber-400">
							<span class="size-1 rounded-full bg-amber-500"></span>
							{$i18n.t('Needs Auth')}
						</span>
					{:else}
						<span class="inline-flex items-center gap-1 rounded-full bg-gray-500/10 px-2 py-0.5 text-xs font-medium text-gray-500 dark:text-gray-400">
							{$i18n.t('Not Installed')}
						</span>
					{/if}

					{#if provider.version}
						<span class="text-xs text-gray-400 dark:text-gray-500">
							{provider.version}
						</span>
					{/if}
				</div>

				<div class="text-xs text-gray-500 dark:text-gray-400 line-clamp-1 mt-0.5">
					{provider.reason || provider.message}
				</div>
			</div>
		</div>

		<div class="flex items-center gap-2">
			{#if provider.id === 'gemini'}
				{#if provider.authenticated}
					<button
						type="button"
						class="flex h-7 items-center justify-center gap-1 rounded-lg border border-gray-200 px-2 text-xs text-gray-600 transition hover:bg-gray-50 disabled:opacity-50 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800"
						disabled={disabled || signingOut}
						on:click={handleSignOut}
					>
						{#if signingOut}
							<Spinner className="size-3" />
						{:else}
							<span>{$i18n.t('Sign out')}</span>
						{/if}
					</button>
				{:else}
					<button
						type="button"
						class="flex h-7 items-center justify-center gap-1 rounded-lg border border-blue-500/30 bg-blue-500/10 px-2.5 text-xs font-medium text-blue-600 transition hover:bg-blue-500/20 disabled:opacity-50 dark:text-blue-400"
						disabled={disabled || startingAuth || !provider.installed}
						on:click={handleStartSignIn}
					>
						{#if startingAuth}
							<Spinner className="size-3" />
							<span>{$i18n.t('Starting...')}</span>
						{:else}
							<span>{$i18n.t('Sign in with Google')}</span>
						{/if}
					</button>
				{/if}
			{/if}

			<button
				type="button"
				class="flex h-7 items-center justify-center gap-1 rounded-lg border border-gray-200 px-2.5 text-xs text-gray-600 transition hover:bg-gray-50 disabled:opacity-50 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800"
				disabled={disabled || testing || !provider.installed || provider.execution_available === false}
				on:click={runTest}
			>
				{#if testing}
					<Spinner className="size-3" />
					<span>{$i18n.t('Testing...')}</span>
				{:else}
					<span>{$i18n.t('Test')}</span>
				{/if}
			</button>

			<Switch
				bind:state={config.enabled}
				disabled={disabled || provider.execution_available === false}
				on:change={() => {
					onUpdate();
				}}
			/>
		</div>
	</div>

	{#if authFlow && authFlow.phase !== 'idle' && authFlow.phase !== 'succeeded'}
		<div class="mt-2 rounded-xl border border-blue-200 bg-blue-50/60 p-3 text-xs dark:border-blue-900/60 dark:bg-blue-950/20">
			{#if authFlow.phase === 'starting'}
				<div class="flex items-center gap-2 text-blue-700 dark:text-blue-300">
					<Spinner className="size-3.5" />
					<span>{$i18n.t('Initializing Google Antigravity sign-in session...')}</span>
				</div>
			{:else if authFlow.phase === 'waiting'}
				<div class="flex flex-col gap-2.5">
					<div class="font-medium text-blue-900 dark:text-blue-200">
						{$i18n.t('Google Account Sign-In')}
					</div>
					<p class="text-xs text-blue-800/80 dark:text-blue-300/80">
						{$i18n.t('Open the link below to sign in with Google. If you are accessing this server remotely, copy the full redirect URL (http://127.0.0.1:.../?code=...) from your browser address bar after signing in and paste it below.')}
					</p>

					{#if authFlow.authorization_url}
						<div class="flex items-center gap-2">
							<a
								href={authFlow.authorization_url}
								target="_blank"
								rel="noopener noreferrer"
								class="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-1.5 font-medium text-white hover:bg-blue-700 transition"
							>
								<span>{$i18n.t('1. Open Google Sign-In')}</span>
							</a>
						</div>
					{/if}

					<div class="flex flex-col gap-1.5 pt-1">
						<label for="gemini-redirect-url" class="font-medium text-gray-700 dark:text-gray-300">
							{$i18n.t('2. Paste redirect URL (remote connection):')}
						</label>
						<div class="flex items-center gap-2">
							<input
								id="gemini-redirect-url"
								type="text"
								class="flex-1 rounded-lg border border-gray-200 bg-white px-2.5 py-1 text-xs dark:border-gray-700 dark:bg-gray-800 dark:text-gray-100"
								placeholder="http://127.0.0.1:45678/?state=...&code=..."
								bind:value={returnUrl}
							/>
							<button
								type="button"
								class="flex h-7 items-center justify-center gap-1 rounded-lg bg-emerald-600 px-3 text-xs font-medium text-white transition hover:bg-emerald-700 disabled:opacity-50"
								disabled={completingAuth || !returnUrl.trim()}
								on:click={handleCompleteSignIn}
							>
								{#if completingAuth}
									<Spinner className="size-3" />
								{:else}
									<span>{$i18n.t('Complete')}</span>
								{/if}
							</button>
							<button
								type="button"
								class="flex h-7 items-center justify-center rounded-lg border border-gray-200 px-2.5 text-xs text-gray-500 hover:bg-gray-50 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-800"
								on:click={handleCancelSignIn}
							>
								{$i18n.t('Cancel')}
							</button>
						</div>
					</div>
				</div>
			{:else if authFlow.phase === 'verifying'}
				<div class="flex items-center gap-2 text-blue-700 dark:text-blue-300">
					<Spinner className="size-3.5" />
					<span>{$i18n.t('Verifying Google account access and discovering models...')}</span>
				</div>
			{:else if authFlow.phase === 'failed'}
				<div class="flex items-center justify-between text-red-600 dark:text-red-400">
					<span>{authFlow.message || $i18n.t('Google sign-in failed.')}</span>
					<button
						type="button"
						class="text-xs underline"
						on:click={handleStartSignIn}
					>
						{$i18n.t('Try again')}
					</button>
				</div>
			{/if}
		</div>
	{/if}

	{#if testResult}
		<div
			class="rounded-lg p-2.5 text-xs {testResult.success
				? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
				: 'bg-red-500/10 text-red-700 dark:text-red-300'}"
		>
			<div class="font-medium">
				{testResult.message}
			</div>
			{#if testResult.response}
				<div class="mt-1 font-mono text-xs text-gray-600 dark:text-gray-400">
					&ldquo;{testResult.response}&rdquo;
				</div>
			{/if}
		</div>
	{/if}

	<div>
		<button
			type="button"
			class="text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 underline"
			on:click={() => (showAdvanced = !showAdvanced)}
		>
			{showAdvanced ? $i18n.t('Hide configuration') : $i18n.t('Advanced configuration')}
		</button>

		{#if showAdvanced}
			<div class="mt-2.5 grid grid-cols-1 gap-2 sm:grid-cols-2 pt-2 border-t border-gray-100 dark:border-gray-800">
				<div>
					<label for={`subscription-${provider.id}-binary`} class="block text-xs text-gray-500 dark:text-gray-400 mb-1">
						{$i18n.t('Binary Path Override')}
					</label>
					<input
						id={`subscription-${provider.id}-binary`}
						type="text"
						class="w-full rounded-lg border border-gray-200 bg-transparent px-2.5 py-1 text-xs dark:border-gray-700 dark:text-gray-100"
						placeholder={provider.binary_path || 'e.g. /home/betheenvy/.local/libexec/t3-antigravity/agy_acp_server.par'}
						bind:value={config.binary_path}
						disabled={disabled}
						on:change={onUpdate}
					/>
				</div>

				<div>
					<label for={`subscription-${provider.id}-config`} class="block text-xs text-gray-500 dark:text-gray-400 mb-1">
						{$i18n.t('Config Directory (Multi-Account / BYOS)')}
					</label>
					<input
						id={`subscription-${provider.id}-config`}
						type="text"
						class="w-full rounded-lg border border-gray-200 bg-transparent px-2.5 py-1 text-xs dark:border-gray-700 dark:text-gray-100"
						placeholder={provider.id === 'claude' ? '~/.claude' : provider.id === 'gemini' ? '~/.gemini-mcchat (GEMINI_HOME)' : '~/.codex'}
						bind:value={config.config_dir}
						disabled={disabled}
						on:change={onUpdate}
					/>
				</div>
			</div>
		{/if}
	</div>
</div>

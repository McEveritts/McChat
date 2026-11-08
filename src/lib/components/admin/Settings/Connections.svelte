<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { createEventDispatcher, onMount, getContext } from 'svelte';

	const dispatch = createEventDispatcher();

	import { getOllamaConfig, updateOllamaConfig } from '$lib/apis/ollama';
	import { getOpenAIConfig, updateOpenAIConfig, getOpenAIModels } from '$lib/apis/openai';
	import { getModels as _getModels, getBackendConfig } from '$lib/apis';
	import { getConnectionsConfig, setConnectionsConfig } from '$lib/apis/configs';

	import { config, models, settings, user } from '$lib/stores';

	import Switch from '$lib/components/common/Switch.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import ArrowPath from '$lib/components/icons/ArrowPath.svelte';
	import Plus from '$lib/components/icons/Plus.svelte';

	import OpenAIConnection from './Connections/OpenAIConnection.svelte';
	import AddConnectionModal from '$lib/components/AddConnectionModal.svelte';
	import OllamaConnection from './Connections/OllamaConnection.svelte';
	import SubscriptionConnection from './Connections/SubscriptionConnection.svelte';
	import AdminSettingRow from './AdminSettingRow.svelte';
	import AdminSettingSection from './AdminSettingSection.svelte';
	import {
		getSubscriptionsConfig,
		updateSubscriptionsConfig,
		getSubscriptionProviders,
		type SubscriptionProviderStatus,
		type SubscriptionsConfig
	} from '$lib/apis/subscriptions';

	const i18n: any = getContext('i18n');

	const getModels = async () => {
		const models = await _getModels(
			localStorage.token,
			$config?.features?.enable_direct_connections ? ($settings?.directConnections ?? null) : null,
			false,
			true
		);
		return models;
	};

	// External
	let OLLAMA_BASE_URLS: string[] = [''];
	let OLLAMA_API_CONFIGS: any = {};

	let OPENAI_API_KEYS: string[] = [''];
	let OPENAI_API_BASE_URLS: string[] = [''];
	let OPENAI_API_CONFIGS: any = {};

	let ENABLE_OPENAI_API: null | boolean = null;
	let ENABLE_OLLAMA_API: null | boolean = null;
	let subscriptionsConfig: SubscriptionsConfig | null = null;
	let subscriptionProviders: SubscriptionProviderStatus[] = [];

	let connectionsConfig: any = null;

	let pipelineUrls: Record<string, boolean> = {};
	let showAddOpenAIConnectionModal = false;
	let showAddOllamaConnectionModal = false;
	let modelListRefreshing = false;
	let savingSubscriptions = false;

	const updateOpenAIHandler = async () => {
		if (ENABLE_OPENAI_API !== null) {
			// Remove trailing slashes
			OPENAI_API_BASE_URLS = OPENAI_API_BASE_URLS.map((url) => url.replace(/\/$/, ''));

			// Check if API KEYS length is same than API URLS length
			if (OPENAI_API_KEYS.length !== OPENAI_API_BASE_URLS.length) {
				// if there are more keys than urls, remove the extra keys
				if (OPENAI_API_KEYS.length > OPENAI_API_BASE_URLS.length) {
					OPENAI_API_KEYS = OPENAI_API_KEYS.slice(0, OPENAI_API_BASE_URLS.length);
				}

				// if there are more urls than keys, add empty keys
				if (OPENAI_API_KEYS.length < OPENAI_API_BASE_URLS.length) {
					const diff = OPENAI_API_BASE_URLS.length - OPENAI_API_KEYS.length;
					for (let i = 0; i < diff; i++) {
						OPENAI_API_KEYS.push('');
					}
				}
			}

			const res = await updateOpenAIConfig(localStorage.token, {
				ENABLE_OPENAI_API: ENABLE_OPENAI_API,
				OPENAI_API_BASE_URLS: OPENAI_API_BASE_URLS,
				OPENAI_API_KEYS: OPENAI_API_KEYS,
				OPENAI_API_CONFIGS: OPENAI_API_CONFIGS
			}).catch((error) => {
				toast.error(`${error}`);
			});

			if (res) {
				toast.success($i18n.t('OpenAI API settings updated'));
				await models.set(await getModels());
			}
		}
	};

	const updateOllamaHandler = async () => {
		if (ENABLE_OLLAMA_API !== null) {
			// Remove trailing slashes
			OLLAMA_BASE_URLS = OLLAMA_BASE_URLS.map((url) => url.replace(/\/$/, ''));

			const res = await updateOllamaConfig(localStorage.token, {
				ENABLE_OLLAMA_API: ENABLE_OLLAMA_API,
				OLLAMA_BASE_URLS: OLLAMA_BASE_URLS,
				OLLAMA_API_CONFIGS: OLLAMA_API_CONFIGS
			}).catch((error) => {
				toast.error(`${error}`);
			});

			if (res) {
				toast.success($i18n.t('Ollama API settings updated'));
				await models.set(await getModels());
			}
		}
	};

	let initialSubscriptionsSnapshot = '';

	let refreshingSubscriptionsAuth = false;
	const refreshSubscriptionsAuthHandler = async () => {
		if (refreshingSubscriptionsAuth) return;
		refreshingSubscriptionsAuth = true;
		try {
			await models.set(await getModels());
			if (
				subscriptionsConfig &&
				subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID &&
				subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID === subscriptionsConfig.current_user_id
			) {
				subscriptionProviders = (await getSubscriptionProviders(localStorage.token)) || [];
			}
		} catch (error) {
			console.error('Failed to refresh subscription auth state/models:', error);
		} finally {
			refreshingSubscriptionsAuth = false;
		}
	};

	const updateSubscriptionsHandler = async () => {
		if (subscriptionsConfig !== null && !savingSubscriptions) {
			if (
				subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID &&
				subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID !== subscriptionsConfig.current_user_id
			) {
				return;
			}
			if (subscriptionsConfig.ENABLE_SUBSCRIPTIONS && !subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID) {
				toast.error($i18n.t('Ownership must be claimed before enabling subscription providers.'));
				subscriptionsConfig.ENABLE_SUBSCRIPTIONS = false;
				return;
			}
			const currentSnapshot = JSON.stringify(subscriptionsConfig);
			if (initialSubscriptionsSnapshot && currentSnapshot === initialSubscriptionsSnapshot) {
				return;
			}
			savingSubscriptions = true;
			const snapshot = JSON.parse(currentSnapshot);
			try {
				const res = await updateSubscriptionsConfig(localStorage.token, snapshot);
				if (res) {
					toast.success($i18n.t('Subscription provider settings updated'));
					await models.set(await getModels());
					if (subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID === subscriptionsConfig.current_user_id) {
						subscriptionProviders = (await getSubscriptionProviders(localStorage.token)) || [];
					}
					subscriptionsConfig = await getSubscriptionsConfig(localStorage.token);
					initialSubscriptionsSnapshot = JSON.stringify(subscriptionsConfig);
				}
			} catch (error) {
				toast.error(`${error}`);
			} finally {
				savingSubscriptions = false;
			}
		}
	};

	const updateConnectionsHandler = async () => {
		const res = await setConnectionsConfig(localStorage.token, connectionsConfig).catch((error) => {
			toast.error(`${error}`);
		});

		if (res) {
			toast.success($i18n.t('Connections settings updated'));
			await models.set(await getModels());
			await config.set(await getBackendConfig());
		}
	};

	const refreshModelListHandler = async () => {
		modelListRefreshing = true;

		try {
			await models.set(await getModels());
			toast.success($i18n.t('Model list refreshed'));
		} catch (error) {
			toast.error(`${error}`);
		} finally {
			modelListRefreshing = false;
		}
	};

	const addOpenAIConnectionHandler = async (connection: any) => {
		OPENAI_API_BASE_URLS = [...OPENAI_API_BASE_URLS, connection.url];
		OPENAI_API_KEYS = [...OPENAI_API_KEYS, connection.key];
		OPENAI_API_CONFIGS[OPENAI_API_BASE_URLS.length - 1] = connection.config;

		await updateOpenAIHandler();
	};

	const addOllamaConnectionHandler = async (connection: any) => {
		OLLAMA_BASE_URLS = [...OLLAMA_BASE_URLS, connection.url];
		OLLAMA_API_CONFIGS[OLLAMA_BASE_URLS.length - 1] = {
			...connection.config,
			key: connection.key
		};

		await updateOllamaHandler();
	};

	onMount(async () => {
		if ($user?.role === 'admin') {
			let ollamaConfig: any = {};
			let openaiConfig: any = {};

			await Promise.all([
				(async () => {
					ollamaConfig = await getOllamaConfig(localStorage.token);
				})(),
				(async () => {
					openaiConfig = await getOpenAIConfig(localStorage.token);
				})(),
				(async () => {
					connectionsConfig = await getConnectionsConfig(localStorage.token);
				})(),
				(async () => {
					try {
						subscriptionsConfig = await getSubscriptionsConfig(localStorage.token);
						if (
							subscriptionsConfig &&
							subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID &&
							subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID === subscriptionsConfig.current_user_id
						) {
							subscriptionProviders = (await getSubscriptionProviders(localStorage.token)) || [];
						} else {
							subscriptionProviders = [];
						}
						initialSubscriptionsSnapshot = JSON.stringify(subscriptionsConfig);
					} catch (e) {
						console.error('Failed to load subscriptions config:', e);
					}
				})()
			]);

			ENABLE_OPENAI_API = openaiConfig.ENABLE_OPENAI_API;
			ENABLE_OLLAMA_API = ollamaConfig.ENABLE_OLLAMA_API;

			OPENAI_API_BASE_URLS = openaiConfig.OPENAI_API_BASE_URLS;
			OPENAI_API_KEYS = openaiConfig.OPENAI_API_KEYS;
			OPENAI_API_CONFIGS = openaiConfig.OPENAI_API_CONFIGS;

			OLLAMA_BASE_URLS = ollamaConfig.OLLAMA_BASE_URLS;
			OLLAMA_API_CONFIGS = ollamaConfig.OLLAMA_API_CONFIGS;

			if (ENABLE_OPENAI_API) {
				// get url and idx
				for (const [idx, url] of OPENAI_API_BASE_URLS.entries()) {
					if (!OPENAI_API_CONFIGS[idx]) {
						// Legacy support, url as key
						OPENAI_API_CONFIGS[idx] = OPENAI_API_CONFIGS[url] || {};
					}
				}

				OPENAI_API_BASE_URLS.forEach(async (url, idx) => {
					OPENAI_API_CONFIGS[idx] = OPENAI_API_CONFIGS[idx] || {};
					if (!(OPENAI_API_CONFIGS[idx]?.enable ?? true)) {
						return;
					}
					const res = await getOpenAIModels(localStorage.token, idx);
					if (res.pipelines) {
						pipelineUrls[url] = true;
					}
				});
			}

			if (ENABLE_OLLAMA_API) {
				for (const [idx, url] of OLLAMA_BASE_URLS.entries()) {
					if (!OLLAMA_API_CONFIGS[idx]) {
						OLLAMA_API_CONFIGS[idx] = OLLAMA_API_CONFIGS[url] || {};
					}
				}
			}
		}
	});

	const submitHandler = async () => {
		await updateOpenAIHandler();
		await updateOllamaHandler();
		await updateSubscriptionsHandler();

		dispatch('save');

		await config.set(await getBackendConfig());
	};
</script>

<AddConnectionModal
	bind:show={showAddOpenAIConnectionModal}
	onSubmit={addOpenAIConnectionHandler}
/>

<AddConnectionModal
	ollama
	bind:show={showAddOllamaConnectionModal}
	onSubmit={addOllamaConnectionHandler}
/>

<form class="flex h-full flex-col justify-between text-sm" on:submit|preventDefault={submitHandler}>
	<h2 class="text-sm font-medium text-gray-900 dark:text-white mb-4">
		{$i18n.t('settings.admin.connections.title')}
	</h2>

	<div class="flex-1 min-h-0 overflow-y-auto scrollbar-hover pr-1.5">
		{#if ENABLE_OPENAI_API !== null && ENABLE_OLLAMA_API !== null && connectionsConfig !== null}
			<AdminSettingSection first>
				<AdminSettingRow label={$i18n.t('settings.admin.connections.openaiApi.label')} let:labelId>
					<Switch
						bind:state={ENABLE_OPENAI_API}
						on:change={async () => {
							updateOpenAIHandler();
						}}
						ariaLabelledbyId={labelId}
					/>
				</AdminSettingRow>

				{#if ENABLE_OPENAI_API}
					<div>
						<div class="mb-2 flex items-center justify-between gap-4">
							<div class="text-xs text-gray-600 dark:text-gray-400">
								{$i18n.t('settings.admin.connections.manageOpenaiApiConnections.label')}
							</div>

							<Tooltip content={$i18n.t('settings.admin.connections.addConnection.label')}>
								<button
									aria-label={$i18n.t('settings.admin.connections.addConnection.label')}
									class="flex size-6 items-center justify-center rounded-lg text-gray-400 transition-colors hover:bg-black/5 hover:text-gray-700 dark:text-gray-600 dark:hover:bg-white/5 dark:hover:text-gray-300"
									on:click={() => {
										showAddOpenAIConnectionModal = true;
									}}
									type="button"
								>
									<Plus />
								</button>
							</Tooltip>
						</div>

						<div class="flex flex-col gap-1.5">
							{#each OPENAI_API_BASE_URLS as url, idx}
								<OpenAIConnection
									bind:url={OPENAI_API_BASE_URLS[idx]}
									bind:key={OPENAI_API_KEYS[idx]}
									bind:config={OPENAI_API_CONFIGS[idx]}
									pipeline={pipelineUrls[url] ? true : false}
									onSubmit={() => {
										updateOpenAIHandler();
									}}
									onDelete={() => {
										OPENAI_API_BASE_URLS = OPENAI_API_BASE_URLS.filter(
											(url, urlIdx) => idx !== urlIdx
										);
										OPENAI_API_KEYS = OPENAI_API_KEYS.filter((key, keyIdx) => idx !== keyIdx);

										let newConfig: any = {};
										OPENAI_API_BASE_URLS.forEach((url, newIdx) => {
											newConfig[newIdx] = OPENAI_API_CONFIGS[newIdx < idx ? newIdx : newIdx + 1];
										});
										OPENAI_API_CONFIGS = newConfig;
										updateOpenAIHandler();
									}}
								/>
							{/each}
						</div>
					</div>
				{/if}

				<AdminSettingRow label={$i18n.t('settings.admin.connections.ollamaApi.label')} let:labelId>
					<Switch
						bind:state={ENABLE_OLLAMA_API}
						on:change={async () => {
							updateOllamaHandler();
						}}
						ariaLabelledbyId={labelId}
					/>
				</AdminSettingRow>

				{#if ENABLE_OLLAMA_API}
					<div>
						<div class="mb-2 flex items-center justify-between gap-4">
							<div class="text-xs text-gray-600 dark:text-gray-400">
								{$i18n.t('settings.admin.connections.manageOllamaApiConnections.label')}
							</div>

							<Tooltip content={$i18n.t('settings.admin.connections.addConnection.label')}>
								<button
									aria-label={$i18n.t('settings.admin.connections.addConnection.label')}
									class="flex size-6 items-center justify-center rounded-lg text-gray-400 transition-colors hover:bg-black/5 hover:text-gray-700 dark:text-gray-600 dark:hover:bg-white/5 dark:hover:text-gray-300"
									on:click={() => {
										showAddOllamaConnectionModal = true;
									}}
									type="button"
								>
									<Plus />
								</button>
							</Tooltip>
						</div>

						<div class="flex flex-col gap-1.5">
							{#each OLLAMA_BASE_URLS as url, idx}
								<OllamaConnection
									bind:url={OLLAMA_BASE_URLS[idx]}
									bind:config={OLLAMA_API_CONFIGS[idx]}
									{idx}
									onSubmit={() => {
										updateOllamaHandler();
									}}
									onDelete={() => {
										OLLAMA_BASE_URLS = OLLAMA_BASE_URLS.filter((url, urlIdx) => idx !== urlIdx);

										let newConfig: any = {};
										OLLAMA_BASE_URLS.forEach((url, newIdx) => {
											newConfig[newIdx] = OLLAMA_API_CONFIGS[newIdx < idx ? newIdx : newIdx + 1];
										});
										OLLAMA_API_CONFIGS = newConfig;
										updateOllamaHandler();
									}}
								/>
							{/each}
						</div>

						<div class="mt-1 text-xs text-gray-400 dark:text-gray-600">
							{$i18n.t('settings.admin.connections.manageOllamaApiConnections.description')}
							<a
								class="font-normal underline hover:text-gray-700 dark:hover:text-gray-300"
								href="https://github.com/open-webui/open-webui#troubleshooting"
								target="_blank"
							>
								{$i18n.t('Click here for help.')}
							</a>
						</div>
					</div>
				{/if}
			</AdminSettingSection>

			{#if subscriptionsConfig !== null}
				<AdminSettingSection
					title={$i18n.t('Subscription Providers (BYOS)')}
				>
					{#if subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID && subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID !== subscriptionsConfig.current_user_id}
						<div class="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-300">
							{$i18n.t('Subscription providers are owned by user')}: <span class="font-mono font-medium">{subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID}</span>. {$i18n.t('Only the registered owner can modify subscription provider settings or view CLI configurations.')}
						</div>
					{:else}
						<AdminSettingRow
							label={$i18n.t('Enable Subscription Providers')}
							description={$i18n.t('Connect official CLI harnesses (Claude Code, OpenAI Codex) with host-owned auth.')}
							let:labelId
						>
							<Switch
								bind:state={subscriptionsConfig.ENABLE_SUBSCRIPTIONS}
								disabled={savingSubscriptions || (!subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID && !subscriptionsConfig.ENABLE_SUBSCRIPTIONS)}
								on:change={async () => {
									if (subscriptionsConfig && !subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID) {
										toast.error($i18n.t('Please claim ownership before enabling subscription providers.'));
										subscriptionsConfig.ENABLE_SUBSCRIPTIONS = false;
										return;
									}
									await updateSubscriptionsHandler();
								}}
								ariaLabelledbyId={labelId}
							/>
						</AdminSettingRow>

						<AdminSettingRow
							label={$i18n.t('Host Subscription Account Owner')}
							description={$i18n.t('Host CLI credentials bind exclusively to an individual user account to strictly comply with provider Terms of Service (no account sharing/intermediation).')}
							let:labelId
						>
							<div class="flex items-center gap-2">
								{#if subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID}
									<span class="rounded bg-gray-100 dark:bg-gray-800 px-2 py-1 text-xs font-mono text-gray-700 dark:text-gray-300">
										{subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID === subscriptionsConfig.current_user_id ? $i18n.t('Your Account (Owner)') : subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID}
									</span>
								{:else}
									<button
										type="button"
										disabled={savingSubscriptions}
										class="rounded-lg bg-emerald-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-emerald-700 transition disabled:opacity-50"
										on:click={async () => {
											if (subscriptionsConfig && subscriptionsConfig.current_user_id) {
												subscriptionsConfig.SUBSCRIPTIONS_OWNER_ID = subscriptionsConfig.current_user_id;
												await updateSubscriptionsHandler();
											}
										}}
									>
										{$i18n.t('Claim Ownership for Current User')}
									</button>
								{/if}
							</div>
						</AdminSettingRow>

						{#if subscriptionsConfig.ENABLE_SUBSCRIPTIONS}
							<div class="mt-2.5 flex flex-col gap-2">
								{#each subscriptionProviders as provider}
									{#if subscriptionsConfig.SUBSCRIPTIONS_PROVIDERS && subscriptionsConfig.SUBSCRIPTIONS_PROVIDERS[provider.id]}
										<SubscriptionConnection
											{provider}
											bind:config={subscriptionsConfig.SUBSCRIPTIONS_PROVIDERS[provider.id]}
										disabled={savingSubscriptions}
											onUpdate={() => {
												updateSubscriptionsHandler();
											}}
											onAuthChanged={() => {
												refreshSubscriptionsAuthHandler();
											}}
										/>
									{/if}
								{/each}
							</div>

							<div class="mt-2 text-xs text-gray-400 dark:text-gray-500">
								{$i18n.t('Authentication is owned by the local provider harnesses. Log in on the server terminal for Claude/Codex, or use Google Sign-in above for Gemini Antigravity. Tokens are never read or stored by McChat.')}
							</div>
						{/if}
					{/if}
				</AdminSettingSection>
			{/if}

			<AdminSettingSection
				title={$i18n.t('settings.admin.connections.sections.userConnections.title')}
			>
				<AdminSettingRow
					label={$i18n.t('settings.admin.connections.directConnections.label')}
					description={$i18n.t('settings.admin.connections.directConnections.description')}
					let:labelId
				>
					<Switch
						bind:state={connectionsConfig.ENABLE_DIRECT_CONNECTIONS}
						on:change={async () => {
							updateConnectionsHandler();
						}}
						ariaLabelledbyId={labelId}
					/>
				</AdminSettingRow>

				<AdminSettingRow
					label={$i18n.t('settings.admin.connections.cacheBaseModelList.label')}
					description={$i18n.t('settings.admin.connections.cacheBaseModelList.description')}
					let:labelId
				>
					<div class="flex items-center gap-1.5">
						{#if connectionsConfig.ENABLE_BASE_MODELS_CACHE}
							<Tooltip content={$i18n.t('settings.admin.connections.refreshModels.label')}>
								<button
									class="flex size-6 items-center justify-center rounded-lg text-gray-400 transition-colors hover:bg-black/5 hover:text-gray-700 disabled:cursor-not-allowed disabled:opacity-50 dark:text-gray-600 dark:hover:bg-white/5 dark:hover:text-gray-300"
									type="button"
									disabled={modelListRefreshing}
									on:click={refreshModelListHandler}
									aria-label={$i18n.t('settings.admin.connections.refreshModels.label')}
								>
									{#if modelListRefreshing}
										<Spinner className="size-3.5" />
									{:else}
										<ArrowPath className="size-4" />
									{/if}
								</button>
							</Tooltip>
						{/if}

						<Switch
							bind:state={connectionsConfig.ENABLE_BASE_MODELS_CACHE}
							on:change={async () => {
								updateConnectionsHandler();
							}}
							ariaLabelledbyId={labelId}
						/>
					</div>
				</AdminSettingRow>
			</AdminSettingSection>
		{:else}
			<div class="flex h-full justify-center">
				<div class="my-auto">
					<Spinner className="size-6" />
				</div>
			</div>
		{/if}
	</div>

	<div class="flex justify-end pt-6 text-sm font-normal">
		<button
			class="flex h-7 shrink-0 items-center justify-center gap-1.5 rounded-lg bg-gray-900 px-2.5 text-xs font-normal text-white transition hover:bg-black disabled:opacity-60 dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-white"
			type="submit"
		>
			{$i18n.t('Save')}
		</button>
	</div>
</form>

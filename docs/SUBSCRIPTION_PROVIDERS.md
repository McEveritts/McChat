# Subscription Provider Connections (Bring Your Own Subscription - BYOS)

McChat supports **Subscription-Provider Connections** alongside standard OpenAI and Ollama API keys, inspired by the Bring Your Own Subscription (BYOS) architecture of [T3 Code](https://github.com/pingdotgg/t3code).

The designated administrator can connect their own **Anthropic Claude Code**, **OpenAI Codex**, and **Google Gemini through Antigravity ACP** accounts. Provider plans, quotas, account eligibility and organization policies still apply. Using an official interface does not imply provider certification of McChat.

---

## 1. Supported Provider Matrix

| Provider | Provider ID | Official CLI Binary | Config Directory Variable | Authentication Method | Isolation Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Anthropic Claude Code** | `claude` | `claude` | `CLAUDE_CONFIG_DIR` | `claude auth login` (OAuth) | `--safe-mode`, `--tools ""`, `--disallowedTools "*"` |
| **OpenAI Codex** | `codex` | `codex` | `CODEX_HOME` | `codex login` (OAuth / ChatGPT) | `--ignore-user-config`, repeatable `--disable`, `--sandbox read-only` |
| **Google Gemini** | `gemini` | `agy_acp_server.par` | `GEMINI_HOME` | Google OAuth (`oauth-personal`) via Admin Connections URL relay | `_meta.antigravity.enabledTools: []`, `fs: false, terminal: false`, dedicated `~/.gemini-mcchat` profile, BROWSER suppression helper |

---

## 2. Architectural & Security Principles

1. **Strict Single-User Host Binding**:
   - Host subscription connections **bind exclusively to one designated administrator owner user**.
   - Cross-user sharing, multi-user pooling, and intermediation are strictly forbidden. There is no multi-user sharing toggle or escape hatch.
   - An administrator claims subscription ownership for their account in Admin Settings. Only this designated account owner can list, configure, test, or converse with subscription models. Non-owners (including other administrators) receive empty model lists, redacted configuration views, or `403 Forbidden` on direct execution.
   - Changing subscription settings requires persistent configuration so ownership can be claimed atomically across workers. With persistent configuration disabled, subscription settings updates return an explicit error before writing the database; environment-provided settings remain readable.
   - Saved subscription presets, configuration exports and version history follow the same owner restriction. Other administrators can continue managing ordinary API/Ollama models, while subscription configuration and rollback history remain private to the subscription owner.

2. **Harness-Owned Authentication**:
   - Authentication is managed and owned entirely by each official provider harness (`claude`, `codex`, `agy_acp_server.par`).
   - McChat **never** reads, extracts, parses, copies, stores, or transmits OAuth access tokens, cookies, refresh tokens, or secret files (`auth.json`, `oauth_creds.json`, etc.).
   - All token lifecycles, OAuth flows, and refreshes remain within each provider's official harness binary.
   - Personal API keys (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`) are actively purged from subprocess execution environments and rejected if detected during status probes.

3. **No Blanket API Equivalence (Conversational Text Only)**:
   - Subscription harnesses are conversational CLI agents, not raw completion or embedding APIs.
   - Subscriptions support conversational text/messages, streaming SSE, and documented reasoning.
   - Subscriptions **do not** support raw completion parameters (custom `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, ceiling limits `max_tokens`/`max_completion_tokens`, structured `response_format`, or custom `stop` sequences). Non-default controls are validated early and rejected with `400 Bad Request`. This includes nested and JSON-valued `custom_params`; effective payloads are checked again after inlet/request filters before middleware side effects.
   - File attachments, multimodal image/audio inputs, custom actions, embeddings, and realtime audio are strictly rejected for subscription models.
   - Built-in host tools (`terminal`, `code_interpreter`, `web_search`, `tools`, `skills`, `files`, `memory`) are disabled and suppressed.

4. **Rigorous Native Tool, MCP, & Extension Isolation**:
   - **Claude Code**:
     - `--safe-mode`: Disables custom plugins, user skills, and user settings.
     - `--tools ""`: Disables built-in host tools (`Bash`, `Edit`, `Read`, `Glob`, `Grep`).
     - `--disallowedTools "*"`: Denies external and MCP tools.
     - `--strict-mcp-config`: Prevents loading local repository MCP configs.
     - `--no-session-persistence`: Prevents session rollout mutation on disk.
   - **OpenAI Codex**:
     - Repeatable `--disable`: Disables `shell_tool`, `hooks`, `plugins`, `apps`, `browser_use`, `computer_use`, `image_generation`, `multi_agent`, `view_image`, `skill_search`, and `code_mode_host`.
     - `--enable skip_host_skill_discovery`: Blocks host skill scanning.
     - `-c project_doc_max_bytes=0`: Neutralizes reading project documentation into context.
     - `--ignore-user-config`: Ignores host `config.toml` while retaining auth.
     - `--ignore-rules`: Ignores host execpolicy `.rules` files.
     - `--sandbox read-only`: Restricts fallback shell execution.
     - `--ephemeral`: Disables session persistence.
     - `--skip-git-repo-check`: Allows running outside Git repositories.
   - **Google Gemini (Antigravity ACP)**:
     - **Agent Client Protocol (ACP) Transport**: Uses official Antigravity ACP JSON-RPC 2.0 communication over standard input/output with the official runtime launcher (`agy_acp_server.par`).
     - **Tool Disablement via `enabledTools`**: Sends `_meta: { antigravity: { enabledTools: [] } }` on `session/new`. In the official runtime (`tool_filter.py`), this explicitly clears all built-in tools (unlike `disabledTools: ["*"]` which does not support wildcards).
     - **Client Capabilities**: Initializes with `fs: { readTextFile: false, writeTextFile: false }` and `terminal: false`, removing all custom filesystem and terminal tools from the session.
     - **Agent Callback Rejection**: Any agent-initiated requests (`fs/*`, `terminal/*`, `permission/*`) are rejected with JSON-RPC error `-32601` (Method not found).
     - **Dedicated Profile Isolation**: Runs within a private, dedicated `GEMINI_HOME` (`~/.gemini-mcchat`, mode `0700`) with `antigravity-acp/settings.json` enforcing `{"auth": {"type": "oauth-personal"}}`. External host MCP configs, hooks, and skills are never linked or loaded.
     - **Headless BROWSER Helper**: Intercepts the OAuth authorization URL to stderr via a preflighted helper script (`__T3_ANTIGRAVITY_AUTH_URL__...`) without opening browsers on headless server hosts.
     - **Environment Purge**: Removes ambient Google API keys (`GEMINI_API_KEY`, `GOOGLE_API_KEY`) and configures `AGY_ACP_FORCE_FILE_STORAGE=1` and `ANTIGRAVITY_HARNESS_PATH` while preserving Whatbox launcher `HOME`.
     - **Runtime Boundary**: The Whatbox launcher’s capability drop and private working directory are not a filesystem sandbox. Tool restrictions rely on the verified native ACP contract; incompatible runtimes must fail closed.
     - **Zero Background Authentication**: Routine status checks and model queries never launch an ACP session or initiate OAuth in the background.

5. **Enterprise & Organization Policy Authority**:
   - Local tools, plugins, and user hooks are neutralized via the invocation flags and protocol isolation detailed above.
   - However, organization-managed provider policies and enterprise restrictions remain authoritative when mandated by enterprise installations.
   - The provider adapters respect host installation boundaries and enterprise-mandated administrative policies without false claims of override.

6. **Prompts via Stdin (No Argv Leakage)**:
   - All provider adapters feed structured prompt text exclusively via standard input (`stdin`).
   - Prompts never appear in `sys.argv` or process listing tables (`ps aux`), preserving confidential chat context on shared host systems.

7. **Bounded Process Lifecycle & Deadlock Prevention**:
   - Every execution runs in an isolated process group (`start_new_session=True`).
   - Concurrency is bounded by an internal semaphore (`MAX_CONCURRENT_SUBSCRIPTIONS`).
   - Total timeouts cover stdin transmission, streaming reads, and process termination.
   - Upon timeout or client cancellation, the entire process group is signaled with `SIGTERM`, granted a bounded grace period, and unconditionally reaped with `SIGKILL` to prevent orphan child processes.
   - Streaming errors are propagated immediately; failed executions never emit a false `finish_reason: "stop"` or `data: [DONE]`.

---

## 3. Server Setup & One-Time Authentication

Claude and Codex use the account owner’s terminal login. Google Antigravity uses the explicit Google sign-in flow in Admin Connections described below. Authentication remains owned by each native runtime.

### Anthropic Claude Code (`claude`)

1. Install Claude Code on the host (e.g. via `npm install -g @anthropic-ai/claude-code`).
2. Log in interactively on the server terminal:
   ```bash
   claude auth login
   ```
   Follow the terminal instructions to complete OAuth authentication in your browser.
3. Verify authentication status:
   ```bash
   claude auth status
   ```
   The output confirms `"authMethod": "claude.ai"`.
4. Config Directory (Optional):
   ```bash
   mkdir -p ~/.claude_custom
   CLAUDE_CONFIG_DIR=~/.claude_custom claude auth login
   ```
   In McChat Admin Settings, specify `~/.claude_custom` in the **Config Directory** field.

### OpenAI Codex (`codex`)

1. Install OpenAI Codex CLI (e.g. `codex` version 0.166.0 or newer).
2. Log in interactively on the server terminal:
   ```bash
   codex login
   ```
   Complete the ChatGPT authentication flow in your browser.
3. Verify authentication status:
   ```bash
   codex login status
   ```
   The output confirms ChatGPT subscription authentication. If an API key is detected, the adapter will report `api_key_detected` and require subscription login.
4. Config Directory (Optional):
   ```bash
   mkdir -p ~/.codex_custom
   CODEX_HOME=~/.codex_custom codex login
   ```
   In McChat Admin Settings, specify `~/.codex_custom` in the **Config Directory** field.

### Google Gemini (`gemini` / Antigravity ACP)

1. **Official Runtime & Launcher**:
   - McChat integrates directly with Google's official Antigravity ACP server runtime (Whatbox operator launcher `/home/betheenvy/.local/libexec/t3-antigravity/agy_acp_server.par` or standard `agy_acp_server` / `agy` in PATH).
   - The launcher wraps the official standalone `.par` package, mounts synthetic sysfs, drops capabilities, and launches with `--uid=`. Standalone `agy` CLI launches with `acp`.
   - Standard `gemini` CLI commands (e.g. headless or `--acp`) are unsupported runtimes; McChat strictly validates official Antigravity executable names.

2. **Web-Based Interactive Sign-In via Admin Settings**:
   - Because McChat uses headless server execution, sign-in does not require running desktop browsers on the server host.
   - Navigate to **Admin Settings** -> **Connections** -> **Subscription Connections** -> **Google Gemini**.
   - Click **Sign In with Google**. McChat launches an ACP authentication session with a preflighted `BROWSER` capture helper.
   - An official Google OAuth URL (`https://accounts.google.com/o/oauth2/v2/auth?...`) is presented. Open this link in your desktop browser.
   - After signing in, your browser redirects to a local loopback callback URL (`http://127.0.0.1:<port>/?state=...&code=...`). Copy this redirect URL from your browser address bar.
   - Paste the redirect URL into the McChat modal and click **Complete Sign-In**. McChat cryptographically validates the state, matches the bound loopback port, and relays the authorization code to complete the OAuth handshake.
   - The flow is bounded by a 300-second timeout and can be cancelled at any time. Flow polling, callback and cancellation must reach the worker which started sign-in: use one backend worker or sticky routing. Public account verification and revocation are shared through bounded, atomic profile metadata. Concurrent native sign-in/logout operations on one profile are rejected until the active operation finishes or is cancelled.

3. **Sign-Out & Session Management**:
   - Click **Sign Out** in Admin Connections to invoke native ACP logout for the isolated profile (`~/.gemini-mcchat`). McChat clears its public verification metadata and reports success only after native logout succeeds. If logout cannot be verified, the UI reports failure and requires revalidation; McChat never deletes token files itself. Already authorized requests may finish within their bounded request deadlines.
   - McChat's isolated profile guarantees that existing T3 Code or system credentials are never modified, read, or deleted.

4. **Config Directory (Optional)**:
   - By default, McChat isolates Gemini subscriptions in `~/.gemini-mcchat` (`GEMINI_HOME`).
   - You can override this in Admin Settings via the **Config Directory** field. Use an empty, dedicated directory. Existing nonempty directories require the McChat ownership marker; global Gemini, T3 and other application profiles are rejected. Malformed settings are preserved and reported for repair. Configuration changes create private, non-clobbering backups.

---

## 4. Supported Models & Catalog Discovery

### Anthropic Claude Code
- `sonnet` (Claude Sonnet - standard intelligence and speed)
- `opus` (Claude Opus - high intelligence and complex tasks)
- `haiku` (Claude Haiku - rapid responses)

### OpenAI Codex
- Default alias: `default` (standard Codex default model)
- Offline bundled catalog discovery: `codex debug models --bundled` supplies models from the installed CLI to production discovery and chat validation. Results are cached for five minutes per runtime file identity; concurrent refreshes share one bounded probe. Discovery failures use the documented compatibility catalog from the verified CLI release for 30 seconds. Configured model overrides take precedence.
- Custom model IDs can be configured in Admin Settings.

### Google Gemini
- **Dynamic Model Discovery**: Models are discovered dynamically from the active, verified Antigravity ACP session (`configOptions` on `session/new`).
- Unverified accounts expose an empty model catalog; McChat never fabricates named default models or plan entitlement claims.
- Model families offering reasoning levels use the native thinking option (`thinking_level` configId, `category="thought_level"`, matched against offered choice values). Choices are recorded only for the model whose options were advertised. Confirmed choices for other models are retained under the same profile/runtime identity and rechecked against each new native session. If the selected model no longer advertises thinking, its previous choices are removed. Customized model names and workspace presets retain native reasoning menus. Personal and per-chat reasoning requests are validated against the selected provider/model; unsupported values fail explicitly.
- Successful Test checks and changed chat catalogs refresh the designated owner’s model picker. Subscription metadata is refreshed independently of the API/Ollama base-model cache.
- Same-account catalog updates merge under a file lock without changing the authentication generation. Initial verification, explicit sign-in and logout advance the generation so work from an earlier account session cannot restore revoked access.
- Context compaction supports subscription summary models and presets. Server-provided API sampling and token ceilings are omitted for these models; explicit unsupported chat controls still fail validation.
- Server-generated titles and summaries use their own text-task context. Source-chat tools, files, active features and terminal metadata are not inherited by subscription task requests. Mixed model comparisons retain API/Ollama server defaults regardless of which model appears first.
- Native streaming resources have independent request deadlines. Paused HTTP consumers cannot keep CLI processes or concurrency slots beyond the request budget; Gemini also closes its native client as soon as its producer completes.
- Configured restrictions in Admin Settings intersect the verified account catalog. If the runtime changes the selected model or requested thinking level during a turn, McChat fails the turn explicitly so saved history cannot attribute an answer to the wrong model. Refresh the catalog, select an available model and retry.

### Account Entitlement vs. Offline Bundled Catalog
- Shipped binary catalogs contain all models bundled with that CLI release, but individual accounts may have plan-specific quota or access tiers (e.g. ChatGPT Plus vs Pro vs Team, or free Google Gemini Code Assist for Individuals vs paid Google One AI Premium Pro/Ultra).
- Entitlements are enforced upstream by each provider. Upstream quota or access rejections fail the request. Public diagnostics use generic error codes to avoid exposing native OAuth URLs, authorization codes, tokens or private paths.

---

## 5. Verification & Test Suites

Tests cover isolated subprocess, protocol, security and frontend contracts. Real Codex streaming/nonstreaming completions and native Antigravity initialization have been verified. Claude account completion and a newly signed-in isolated Google account completion remain unverified; mocked tests do not establish those account checks. Full browser and application-build validation are separate from the standalone Svelte compiler checks.

```bash
# Run all Python subscription test suites (Gemini, Codex, Adapters, Core, Integration)
./.venv/bin/pytest test/test_subscription*.py

# Run Node.js frontend and API contract test suite
node test/subscriptions-api.test.mjs
```

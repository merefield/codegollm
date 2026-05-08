# CodeGollm

`codegollm` is a minimal Go/Bubble Tea coding TUI. It supports OpenAI API-key auth, ChatGPT OAuth via the ChatGPT/Codex Responses backend, and local Ollama models.

It exposes four model tools:

- `read`
- `write`
- `edit`
- `bash`

File tools are constrained to the directory where `codegollm` is launched, or below it. Bash commands run with that directory as their working directory and require approval.

Prompt controls:

- `Enter`: send the current prompt.
- `Up` / `Down`: navigate prompts sent during the current run.
- `Ctrl+U`: clear the current prompt.
- `Esc`: interrupt an active model response.
- `Ctrl+C`: quit.

While a model call, tool run, model list, or login flow is active, the `working...` indicator animates.

Approval controls:

- `y` or `Enter`: approve once.
- `a`: always approve this exact operation and save that approval in `config.yaml`.
- `n` or `Esc`: deny and type revised instructions for what codegollm should do differently.

Slash commands:

- `/auth` shows configured auth profiles and the auth store path.
- `/login openai-api-key [profile] [env]` uses an OpenAI API key from an environment variable.
- `/login openai-codex [profile] [workspace_id]` logs in with a ChatGPT account and stores a local auth profile.
- `/logout [profile]` removes an auth profile.
- `/model` lists a curated, scrollable set of available models for the active provider. OpenAI API-key auth calls `/v1/models` and filters out non-chat models. ChatGPT auth uses a built-in Codex-compatible model list.
- `/model all` opens the full scrollable model list.
- `/model auto` saves `model: auto`, so each OpenAI/ChatGPT request uses the first available entry from `preferred_models`.
- `/reasoning [none|minimal|low|medium|high|xhigh]` shows or saves the OpenAI reasoning level.
- `/fast [on|off|toggle]` toggles fast model preference for automatic model selection.
- `/help` lists available commands.

## Requirements

- Go 1.22+
- An OpenAI API key in your shell environment, or a ChatGPT login through `/login openai-codex`
- A tool-capable model, defaulting to `gpt-4.1-mini` for OpenAI API-key auth

```bash
export OPENAI_API_KEY=sk-...
```

You can also put it in a `.env` file in the workspace where you launch `codegollm`, or beside the `codegollm` binary:

```bash
OPENAI_API_KEY=sk-...
```

Do not store the API key in `config.yaml`. The config stores the environment variable name:

```yaml
provider: openai
model: gpt-4.1-mini
auth_profile: openai-api-key:default
preferred_models:
  - gpt-5.5
  - gpt-5.4
  - gpt-5.4-mini
  - gpt-5.2-codex
  - gpt-5.1-codex
  - gpt-5-codex
  - gpt-5
  - gpt-4.1
  - gpt-4.1-mini
reasoning_level: medium
fast: false
openai_base_url: https://api.openai.com/v1
openai_api_key_env: OPENAI_API_KEY
```

Use `.env.example` as the template. Keep real `.env` files uncommitted.

Alternatively, run `/login openai-codex` inside the TUI to authenticate with a ChatGPT account. CodeGollm opens a browser OAuth flow and stores tokens in `~/.config/codegollm/auth-profiles.json` with file mode `0600`. The workspace `config.yaml` only stores the selected profile id and optional workspace id:

```yaml
provider: openai-codex
auth_profile: openai-codex:default
chatgpt_workspace_id: org-...
```

ChatGPT login uses the ChatGPT/Codex Responses backend with the OAuth access token. If the optional generated API-key exchange fails, CodeGollm keeps the ChatGPT profile and uses Responses directly. The login success message reports this as `using ChatGPT Responses backend`.

For Ollama, change the provider and model:

```yaml
provider: ollama
model: gemma4:latest
ollama_url: http://localhost:11434
```

## Run

```bash
go run .
```

To rebuild the local binary:

```bash
go build -o codegollm .
```

Edit `config.yaml` to change the model, Ollama URL, or system prompt. Shell flags can override runtime behaviour without saving config:

```bash
codegollm --reasoning high --fast
codegollm --reasoning low --no-fast
```

Set `model: auto` to choose the first available entry from `preferred_models` at request time. With an explicit model name, CodeGollm uses that model until you change it. The `/model` picker also uses `preferred_models` to mark and order the recommended OpenAI/ChatGPT choice.

`reasoning_level` is sent as OpenAI `reasoning_effort` for reasoning-capable API-key models, and as Responses `reasoning.effort` for ChatGPT/Codex Responses calls. It is omitted for non-reasoning models such as GPT-4.1. `none` is omitted for older reasoning models that do not support it, and unsupported `xhigh` requests are downgraded to `high`. `fast: true` biases automatic model selection toward `mini` or `nano` models rather than sending provider-specific latency controls that may not be available on every account.

`recent_history_messages` controls how many recent chat/tool messages are sent verbatim. Older messages are folded into a running summary so the agent keeps durable context without sending the whole session every turn. The default is `10`. Tool output is capped before it is stored and capped more tightly when sent back as model context. If a selected model still reports a context-window error, CodeGollm retries once with a compact recent-history tail.

`approved_tools` stores exact approved operations. Bash approvals are exact command strings. Write and edit approvals include the exact target path and proposed content/change.

Each workspace gets a resumable session file at `.codegollm/session.json`. On startup, `codegollm` loads that file if present and resumes the previous conversation summary and message history.

When launched from another directory, `codegollm` uses that directory as the workspace root. It loads `config.yaml` from the current directory if present, otherwise from the directory containing the `codegollm` binary.

## License

CodeGollm is licensed under the GNU General Public License, version 2. See `LICENSE.txt` and `COPYRIGHT.txt`.

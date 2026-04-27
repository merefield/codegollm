# CodeGollm

`codegollm` is a minimal Go/Bubble Tea coding TUI backed by OpenAI Chat Completions by default, with optional Ollama support.

It exposes four model tools:

- `read`
- `write`
- `edit`
- `bash`

File tools are constrained to the directory where `codegollm` is launched, or below it. Bash commands run with that directory as their working directory and require approval.

Approval controls:

- `y` or `Enter`: approve once.
- `a`: always approve this exact operation and save that approval in `config.yaml`.
- `n` or `Esc`: deny and type revised instructions for what codegollm should do differently.

Slash commands:

- `/model` lists installed Ollama models from `ollama list` when `provider: ollama` is configured.
- `/help` lists available commands.

## Requirements

- Go 1.22+
- An OpenAI API key in your shell environment
- A tool-capable model, defaulting to `gpt-4.1-mini`

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
openai_base_url: https://api.openai.com/v1
openai_api_key_env: OPENAI_API_KEY
```

Use `.env.example` as the template. Keep real `.env` files uncommitted.

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

Edit `config.yaml` to change the model, Ollama URL, or system prompt.

`recent_history_messages` controls how many recent chat/tool messages are sent verbatim. Older messages are folded into a running summary so the agent keeps durable context without sending the whole session every turn. The default is `10`.

`approved_tools` stores exact approved operations. Bash approvals are exact command strings. Write and edit approvals include the exact target path and proposed content/change.

Each workspace gets a resumable session file at `.codegollm/session.json`. On startup, `codegollm` loads that file if present and resumes the previous conversation summary and message history.

When launched from another directory, `codegollm` uses that directory as the workspace root. It loads `config.yaml` from the current directory if present, otherwise from the directory containing the `codegollm` binary.

## License

CodeGollm is licensed under the GNU General Public License, version 2. See `LICENSE.txt` and `COPYRIGHT.txt`.

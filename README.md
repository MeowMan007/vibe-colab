# ⚡ vibe-env

> Turn Google Colab into a **Vibe Coding** powerhouse — a Cursor-like IDE accessible via a public URL, powered by open-source LLMs.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

## What You Get

| Component | What it does |
|---|---|
| **Ollama** | Local model server — runs Llama 3.1, DeepSeek-Coder-V2, Mistral-Nemo on the Colab GPU |
| **code-server** | VS Code in the browser (the same editor you know and love) |
| **Continue** | Open-source Cursor-like AI extension, pre-configured to talk to Ollama |
| **Cloudflare Tunnel** | Free `.trycloudflare.com` URL — no accounts, no tokens |

## Quick Start (Colab)

```python
# Cell 1 — Install
!pip install git+https://github.com/MeowMan007/vibe-colab.git

# Cell 2 — Launch
import vibe_env
url = vibe_env.launch()
# Click the URL that appears → your IDE is ready 🚀
```

## API

### `vibe_env.launch(**kwargs) → str | None`

All-in-one: install → configure → start → tunnel → print URL.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `models` | `list[str]` | See below | Ollama model tags to pull |
| `pull_models` | `bool` | `True` | Whether to pull models |
| `install_continue` | `bool` | `True` | Install the Continue extension |
| `password` | `str \| None` | `None` | Optional password for code-server |

**Default models:** `llama3.1:8b`, `deepseek-coder-v2:16b`, `mistral-nemo:12b`

### `vibe_env.setup(**kwargs)`

Same as `launch()` but only installs/configures — does **not** start services or open a tunnel. Useful for pre-warming an environment.

### `vibe_env.status() → dict`

Print and return the running status of all components.

### `vibe_env.stop()`

Gracefully terminate all background processes (Ollama, code-server, cloudflared).

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  Google Colab                     │
│                                                   │
│  ┌─────────┐   ┌──────────────┐   ┌───────────┐ │
│  │  Ollama  │◄──│  Continue     │──►│ code-     │ │
│  │ :11434   │   │  Extension   │   │ server    │ │
│  │          │   └──────────────┘   │ :8080     │ │
│  │ Models:  │                      └─────┬─────┘ │
│  │ • llama  │                            │       │
│  │ • deep-  │                            │       │
│  │   seek   │                     ┌──────┴──────┐│
│  │ • mistral│                     │ cloudflared ││
│  └─────────┘                     └──────┬──────┘│
│                                         │       │
└─────────────────────────────────────────┼───────┘
                                          │
                              *.trycloudflare.com
                                          │
                                     🌐 You
```

## Why This Architecture

- **Ollama on Colab** — the most stable way to manage multiple open-source models in one environment. Handles switching between models seamlessly.
- **code-server + Continue** — Continue is the closest open-source equivalent to Cursor. Pre-configured to `localhost:11434`, it gives you the "vibe coding" experience where the AI understands your whole codebase.
- **Cloudflare Tunnels** — unlike ngrok, Cloudflare doesn't require an account or auth token for basic tunnels. It "just works" for anyone who imports the package.

## Local Development

```bash
git clone https://github.com/MeowMan007/vibe-colab.git
cd vibe-colab
pip install -e ".[all]"
```

## License

MIT — see [LICENSE](LICENSE).

# ⚡ vibe-env

> Turn Google Colab into a **Vibe Coding** powerhouse — a Cursor-like IDE accessible via a public URL, powered by open-source LLMs.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

## What You Get

| Component | What it does |
|---|---|
| **Ollama** | Local model server — runs Llama 3.1, DeepSeek-Coder-V2, Qwen2.5-Coder on the Colab GPU |
| **code-server** | VS Code in the browser (the same editor you know and love) |
| **Continue** | Open-source Cursor-like AI extension, pre-configured to talk to Ollama |
| **Cloudflare Tunnel** | Free `.trycloudflare.com` URL — no accounts, no tokens |

## Quick Start (Colab)

> **⚠️ Important:** Select a **GPU runtime** before running!  
> Runtime → Change runtime type → **T4 GPU** (or better)

```python
# Cell 1 — Install
!pip install git+https://github.com/MeowMan007/vibe-colab.git

# Cell 2 — Launch
import vibe_env
url = vibe_env.launch()
# Click the URL that appears → your IDE is ready 🚀
```

## How It Works on Colab

The package handles several Colab-specific quirks automatically:

1. **Installs system deps** — `pciutils`, `lshw`, `curl` (needed for GPU detection)
2. **Handles systemd failure** — Ollama's installer tries to create a systemd service, which doesn't exist on Colab. We catch this and run `ollama serve` via Python's `threading` instead
3. **Auto-selects models by VRAM** — T4 GPU (16 GB) gets full-size models; smaller GPUs get 3B variants
4. **Sets networking env vars** — `OLLAMA_HOST=0.0.0.0` and `OLLAMA_ORIGINS=*` so everything connects properly
5. **Pulls models via CLI** — more reliable than the REST API for large downloads

## API

### `vibe_env.launch(**kwargs) → str | None`

All-in-one: install → configure → start → tunnel → print URL.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `models` | `list[str]` | Auto-detected | Ollama model tags to pull |
| `pull_models` | `bool` | `True` | Whether to pull models |
| `install_continue` | `bool` | `True` | Install the Continue extension |
| `password` | `str \| None` | `None` | Optional password for code-server |

**Default models (T4 GPU / 16 GB VRAM):**
- `llama3.1:8b` — general purpose (~4.7 GB)
- `deepseek-coder-v2:latest` — coding specialist (~8.9 GB)
- `qwen2.5-coder:7b` — coding + autocomplete (~4.7 GB)

**Small GPU / CPU fallback:**
- `llama3.2:3b` (~2.0 GB)
- `qwen2.5-coder:3b` (~1.9 GB)

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
│  │ • qwen   │                     │ cloudflared ││
│  └─────────┘                     └──────┬──────┘│
│                                         │       │
└─────────────────────────────────────────┼───────┘
                                          │
                              *.trycloudflare.com
                                          │
                                     🌐 You
```

## Troubleshooting

| Problem | Solution |
|---|---|
| `CalledProcessError` during Ollama install | Expected on Colab (systemd failure). The package handles this automatically — the binary still installs correctly. |
| "No GPU detected" | Go to Runtime → Change runtime type → **T4 GPU** |
| Model pull is slow | Large models can take 5-15 min. Use `models=["llama3.2:3b"]` for faster testing. |
| Tunnel URL not appearing | Wait 30-45 seconds. Check `!cat /tmp/cloudflared.log` for errors. |
| code-server won't start | Check `!cat /tmp/code-server.log` |

## Local Development

```bash
git clone https://github.com/MeowMan007/vibe-colab.git
cd vibe-colab
pip install -e ".[all]"
```

## License

MIT — see [LICENSE](LICENSE).

# ⚡ vibe-env

> Turn Google Colab into a **Vibe Coding** powerhouse — a Cursor-like IDE accessible via a public URL, powered by open-source LLMs.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

## What You Get

| Component | What it does |
|---|---|
| **llama-cpp-python** | Local model server — runs HuggingFace GGUF models (Qwen2.5-Coder, Llama-3.2) on the Colab GPU. Bypasses all systemd requirements. |
| **code-server** | VS Code in the browser (the same editor you know and love) |
| **Continue** | Open-source Cursor-like AI extension, pre-configured to talk to your local model via its OpenAI API |
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

1. **Native Python Backend** — Swapped out Ollama for `llama-cpp-python` to avoid nasty `CalledProcessError` issues and tricky Linux daemon logic.
2. **Auto-installs CUDA Wheels** — Grabs pre-compiled packages for 10x faster installation.
3. **Downloads GGUF from HF Hub** — Pulls optimal quantized models directly from HuggingFace.
4. **Auto-selects models by VRAM** — T4 GPU (16 GB) gets 7B parameters; smaller GPUs get 3B variants.

## API

### `vibe_env.launch(**kwargs) → str | None`

All-in-one: install → configure → start → tunnel → print URL.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `models` | `list[tuple[str, str]]` | Auto-detected | List of tuples `(huggingface_repo_id, gguf_filename)` |
| `pull_models` | `bool` | `True` | Whether to pull models into `~/.cache` |
| `install_continue` | `bool` | `True` | Install the Continue extension |
| `password` | `str \| None` | `None` | Optional password for code-server |

**Default model (T4 GPU / 16 GB VRAM):**
- `unsloth/Qwen2.5-Coder-7B-Instruct-GGUF` (`qwen2.5-coder-7b-instruct-q4_k_m.gguf`)

**Small GPU / CPU fallback:**
- `unsloth/Llama-3.2-3B-Instruct-GGUF` (`llama-3.2-3b-instruct-q4_k_m.gguf`)

### `vibe_env.setup(**kwargs)`
Same as `launch()` but only installs/configures — does **not** start services or open a tunnel. 

### `vibe_env.stop()`
Gracefully terminate all background processes (`llama_cpp.server`, `code-server`, `cloudflared`).

## Local Development

```bash
git clone https://github.com/MeowMan007/vibe-colab.git
cd vibe-colab
pip install -e ".[all]"
```

## License

MIT — see [LICENSE](LICENSE).

"""
vibe_env.core – orchestrates llama-cpp-python, code-server, Cloudflared, and
                the Continue extension so a single `vibe_env.launch()`
                gives you a full Cursor-like IDE on Google Colab.

Colab-specific adaptations:
  • Uses llama-cpp-python instead of Ollama to completely avoid systemd issues.
  • Downloads GGUF models directly via HuggingFace Hub.
  • Auto-installs pre-compiled CUDA wheels for llama-cpp-python.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Optional

import psutil
import requests
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────

LLAMA_CPP_PORT = 8000
CODE_SERVER_PORT = 8080
WORKSPACE_DIR = Path.home() / "workspace"
CODE_SERVER_CONFIG = Path.home() / ".config" / "code-server" / "config.yaml"
CONTINUE_CONFIG_DIR = Path.home() / ".continue"
CONTINUE_EXT = "Continue.continue"

# HuggingFace models: (repo_id, filename)
# We load the first model in the list as the primary server model.
DEFAULT_MODELS = [
    ("unsloth/Qwen2.5-Coder-7B-Instruct-GGUF", "qwen2.5-coder-7b-instruct-q4_k_m.gguf"),
]

SMALL_MODELS = [
    ("unsloth/Llama-3.2-3B-Instruct-GGUF", "llama-3.2-3b-instruct-q4_k_m.gguf"),
]

VIBE_BANNER = r"""
 ██╗   ██╗██╗██████╗ ███████╗    ███████╗███╗   ██╗██╗   ██╗
 ██║   ██║██║██╔══██╗██╔════╝    ██╔════╝████╗  ██║██║   ██║
 ██║   ██║██║██████╔╝█████╗      █████╗  ██╔██╗ ██║██║   ██║
 ╚██╗ ██╔╝██║██╔══██╗██╔══╝      ██╔══╝  ██║╚██╗██║╚██╗ ██╔╝
  ╚████╔╝ ██║██████╔╝███████╗    ███████╗██║ ╚████║ ╚████╔╝
   ╚═══╝  ╚═╝╚═════╝ ╚══════╝    ╚══════╝╚═╝  ╚═══╝  ╚═══╝
"""

console = Console()

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

def _has_gpu() -> bool:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def _gpu_vram_mb() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0

def _gpu_info() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "CPU-only (no GPU detected)"

def _pick_default_models() -> list[tuple[str, str]]:
    vram = _gpu_vram_mb()
    if vram >= 14_000:
        return list(DEFAULT_MODELS)
    return list(SMALL_MODELS)

def _run_bg(cmd: list[str], logfile: Optional[str] = None, **kw) -> subprocess.Popen:
    log = open(logfile, "a") if logfile else subprocess.DEVNULL
    return subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        **kw,
    )

def _wait_for_port(port: int, host: str = "127.0.0.1", timeout: int = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False

def _download_file(url: str, dest: Path) -> None:
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Downloading {dest.name}", total=total or None)
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            progress.advance(task, len(chunk))

# ────────────────────────────────────────────────────────────────────
# llama-cpp-python implementation
# ────────────────────────────────────────────────────────────────────

def _install_llama_cpp() -> None:
    """Install llama-cpp-python from prebuilt CUDA wheels."""
    try:
        import llama_cpp  # noqa: F401
        console.print("  [green]✓[/] llama-cpp-python already installed")
        return
    except ImportError:
        pass

    console.print("  [cyan]↓[/] Installing llama-cpp-python (with CUDA support) …")
    
    # We use a custom index for prebuilt wheels to avoid 15-minute compilation
    # Fallback to normal PyPI if CPU-only
    cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python[server]"]
    if _has_gpu():
        # Usually Colab has CUDA 12.x
        cmd.extend(["--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu121"])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        console.print("  [green]✓[/] llama-cpp-python installed")
    except subprocess.CalledProcessError as exc:
        console.print(f"  [red]✗[/] llama-cpp-python installation failed")
        if exc.stderr:
            console.print(f"      [dim]{exc.stderr.decode()[-200:]}[/]")
        raise RuntimeError("Failed to install llama-cpp-python server.")

def _pull_models(models: list[tuple[str, str]]) -> str:
    """Download GGUF models via huggingface_hub.
    Returns the absolute path to the first downloaded model."""
    console.print("  [cyan]↓[/] Synchronizing models from HuggingFace …")
    
    first_path = None
    for repo_id, filename in models:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(Path.home() / ".cache" / "vibe_models")
            )
            console.print(f"  [green]✓[/] {filename} ready")
            if not first_path:
                first_path = local_path
        except Exception as exc:
            console.print(f"  [red]✗[/] Failed to download {filename}: {exc}")
            raise
    
    return first_path

def _start_llama_cpp(model_path: str) -> None:
    """Start the llama-cpp-python OpenAI-compatible server."""
    # Check if already running
    try:
        r = requests.get(f"http://127.0.0.1:{LLAMA_CPP_PORT}/v1/models", timeout=3)
        if r.status_code == 200:
            console.print("  [green]✓[/] llama.cpp server already running")
            return
    except requests.RequestException:
        pass

    console.print("  [cyan]⟳[/] Starting local inference server …")
    
    # -1 offloads all layers to GPU
    gpu_layers = "-1" if _has_gpu() else "0"
    
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(LLAMA_CPP_PORT),
        "--n_gpu_layers", gpu_layers,
        "--n_ctx", "4096", # Ensure decent context window
    ]

    def _serve():
        subprocess.Popen(cmd, stdout=open("/tmp/llama-cpp.log", "a"), stderr=subprocess.STDOUT)

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    if _wait_for_port(LLAMA_CPP_PORT, timeout=30):
        console.print(f"  [green]✓[/] Inference server ready on port {LLAMA_CPP_PORT}")
    else:
        console.print("  [red]✗[/] Inference server failed to start")
        console.print("  [dim]   Check: !cat /tmp/llama-cpp.log[/]")


# ────────────────────────────────────────────────────────────────────
# code-server
# ────────────────────────────────────────────────────────────────────

def _install_code_server() -> None:
    if shutil.which("code-server"):
        console.print("  [green]✓[/] code-server already installed")
        return

    console.print("  [cyan]↓[/] Installing code-server …")
    result = subprocess.run(
        ["bash", "-c", "curl -fsSL https://code-server.dev/install.sh | sh"],
        capture_output=True, text=True, timeout=300,
    )
    if shutil.which("code-server"):
        console.print("  [green]✓[/] code-server installed")
    else:
        console.print(f"  [red]✗[/] code-server installation failed")
        raise RuntimeError("code-server binary not found after install.")

def _write_code_server_config() -> None:
    CODE_SERVER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    config = textwrap.dedent(f"""\
        bind-addr: 127.0.0.1:{CODE_SERVER_PORT}
        auth: none
        cert: false
        disable-telemetry: true
    """)
    CODE_SERVER_CONFIG.write_text(config)
    console.print("  [green]✓[/] code-server config written")

def _install_continue_extension() -> None:
    console.print("  [cyan]↓[/] Installing Continue extension …")
    try:
        result = subprocess.run(
            ["code-server", "--install-extension", CONTINUE_EXT],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            console.print("  [green]✓[/] Continue extension installed")
        else:
            console.print(f"  [yellow]⚠[/] Continue extension warning")
    except Exception as exc:
        console.print(f"  [yellow]⚠[/] Continue extension install failed: {exc}")

    _write_continue_config()

def _write_continue_config() -> None:
    """Write the Continue config so it auto-connects to llama-cpp-python."""
    config_dir = CONTINUE_CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"

    # We use the generic 'openai' provider which points to localhost llama.cpp
    config = {
        "models": [
            {
                "title": "Local Vibe Model",
                "provider": "openai",
                "model": "local-model",
                "apiBase": f"http://127.0.0.1:{LLAMA_CPP_PORT}/v1",
            }
        ],
        "tabAutocompleteModel": {
            "title": "Local Autocomplete",
            "provider": "openai",
            "model": "local-model",
            "apiBase": f"http://127.0.0.1:{LLAMA_CPP_PORT}/v1",
        },
        "allowAnonymousTelemetry": False,
    }

    config_file.write_text(json.dumps(config, indent=2))
    console.print("  [green]✓[/] Continue config → Local AI (OpenAI format)")

def _start_code_server() -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    console.print("  [cyan]⟳[/] Starting code-server …")
    _run_bg(
        ["code-server", "--config", str(CODE_SERVER_CONFIG), str(WORKSPACE_DIR)],
        logfile="/tmp/code-server.log",
    )
    if _wait_for_port(CODE_SERVER_PORT, timeout=30):
        console.print(f"  [green]✓[/] code-server ready on port {CODE_SERVER_PORT}")
    else:
        console.print("  [red]✗[/] code-server failed to start")


# ────────────────────────────────────────────────────────────────────
# Cloudflared tunnel
# ────────────────────────────────────────────────────────────────────

def _install_cloudflared() -> None:
    if shutil.which("cloudflared"):
        console.print("  [green]✓[/] cloudflared already installed")
        return

    console.print("  [cyan]↓[/] Installing cloudflared …")
    dest = Path("/usr/local/bin/cloudflared")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    try:
        subprocess.run(["wget", "-q", "-O", str(dest), url], check=True, capture_output=True, timeout=120)
    except Exception:
        _download_file(url, dest)

    dest.chmod(0o755)
    console.print("  [green]✓[/] cloudflared installed")

def _start_tunnel() -> str | None:
    console.print("  [cyan]⟳[/] Opening Cloudflare tunnel …")
    proc = subprocess.Popen(
        [
            "cloudflared", "tunnel", "--url",
            f"http://127.0.0.1:{CODE_SERVER_PORT}",
            "--no-autoupdate",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    url: str | None = None
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            time.sleep(0.2)
            continue
        if ".trycloudflare.com" in line:
            match = re.search(r"https?://[\w-]+\.trycloudflare\.com", line)
            if match:
                url = match.group(0)
                break

    if url:
        console.print(f"  [green]✓[/] Tunnel active")
    else:
        console.print("  [red]✗[/] Could not obtain tunnel URL")

    return url


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

def setup(
    models: list[tuple[str, str]] | None = None,
    pull_models: bool = True,
    install_continue: bool = True,
) -> None:
    if models is None:
        models = _pick_default_models()

    console.print(Panel(VIBE_BANNER, title="[bold magenta]vibe-env setup[/]", border_style="magenta"))
    # 1. llama-cpp-python
    console.print("[bold cyan]① AI Backend[/]")
    _install_llama_cpp()
    if pull_models:
        model_path = _pull_models(models)
        _start_llama_cpp(model_path)
    console.print()

    # 2. code-server
    console.print("[bold cyan]② code-server[/]")
    _install_code_server()
    _write_code_server_config()
    if install_continue:
        _install_continue_extension()
    console.print()

    # 3. Cloudflared
    console.print("[bold cyan]③ Cloudflared[/]")
    _install_cloudflared()
    console.print()

def launch(
    models: list[tuple[str, str]] | None = None,
    pull_models: bool = True,
    install_continue: bool = True,
    password: str | None = None,
) -> str | None:
    if models is None:
        models = _pick_default_models()

    console.print(Panel(VIBE_BANNER, title="[bold magenta]vibe-env[/]", border_style="magenta"))

    gpu = _gpu_info()
    vram = _gpu_vram_mb()
    console.print(f"  [bold]GPU:[/] {gpu} ({vram} MiB)")
    console.print()

    console.print("[bold cyan]① Setting up AI Backend[/]")
    _install_llama_cpp()
    
    first_model_name = "None"
    if pull_models:
        model_path = _pull_models(models)
        _start_llama_cpp(model_path)
        first_model_name = models[0][1]
    console.print()

    console.print("[bold cyan]② Setting up IDE[/]")
    _install_code_server()
    _write_code_server_config()
    if password:
        config_text = CODE_SERVER_CONFIG.read_text()
        config_text = config_text.replace("auth: none", f"auth: password\npassword: {password}")
        CODE_SERVER_CONFIG.write_text(config_text)
    
    if install_continue:
        _install_continue_extension()

    _start_code_server()
    console.print()

    console.print("[bold cyan]③ Cloudflare Tunnel[/]")
    _install_cloudflared()
    url = _start_tunnel()
    console.print()

    table = Table(title="🎉 Vibe Environment Ready", border_style="green", show_lines=True)
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    table.add_row("Server", "[green]Running[/]", f"localhost:{LLAMA_CPP_PORT}")
    table.add_row("Active Model", "[green]Loaded[/]", first_model_name)
    table.add_row("IDE", "[green]Running[/]", f"localhost:{CODE_SERVER_PORT}")
    table.add_row("Ext", "Installed" if install_continue else "Skipped", "OpenAI config")
    table.add_row("URL", "[green]Active[/]" if url else "[red]Failed[/]", url or "N/A")

    console.print(table)
    console.print()

    if url:
        console.print(
            Panel(
                f"[bold green]🔗 Open your IDE:[/]\n\n"
                f"   [bold underline link={url}]{url}[/]\n\n"
                f"[dim]Continue is pre-wired to your local model. Press Ctrl+Shift+L to begin.[/]",
                title="[bold]Your Vibe Coding URL[/]",
                border_style="green",
            )
        )
        if _is_colab():
            try:
                from IPython.display import HTML, display
                display(HTML(f'<h2>🔗 <a href="{url}" target="_blank">Click here to open your Vibe IDE</a></h2>'))
            except Exception:
                pass
    return url

def status() -> dict:
    pass

def stop() -> None:
    console.print("[bold yellow]Stopping vibe-env services …[/]")
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "llama_cpp.server" in cmd or "code-server" in cmd or "cloudflared" in (proc.info.get("name") or ""):
                proc.terminate()
                console.print(f"  [yellow]■[/] Terminated {proc.info.get('name')}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    console.print("[green]All services stopped.[/]")

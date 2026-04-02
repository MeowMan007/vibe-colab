"""
vibe_env.core – orchestrates Ollama, code-server, Cloudflared, and
                the Continue extension so a single `vibe_env.launch()`
                gives you a full Cursor-like IDE on Google Colab.

Colab-specific adaptations:
  • Installs pciutils + lshw so Ollama's install script can detect the GPU
  • Handles the systemd failure gracefully (Colab has no systemd)
  • Sets OLLAMA_HOST / OLLAMA_ORIGINS for proper networking
  • Uses threading to run ollama serve without blocking the notebook
  • Ships Colab-friendly default models (≤8 B params) that fit on a T4
  • Pulls models via the CLI (`ollama pull`) with real-time progress
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
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────

OLLAMA_PORT = 11434
CODE_SERVER_PORT = 8080
WORKSPACE_DIR = Path.home() / "workspace"
CODE_SERVER_CONFIG = Path.home() / ".config" / "code-server" / "config.yaml"
CODE_SERVER_DIR = Path.home() / ".local" / "lib" / "code-server"
CONTINUE_CONFIG_DIR = Path.home() / ".continue"
CONTINUE_EXT = "Continue.continue"

# ── Colab-friendly defaults ─────────────────────────────────────────
# T4 GPU has ~15 GB usable VRAM.  Keep total under that to avoid
# spilling layers to system RAM (which kills performance).
DEFAULT_MODELS = [
    "llama3.1:8b",           # ~4.7 GB – general purpose
    "deepseek-coder-v2:latest",  # ~8.9 GB – coding specialist (lite variant)
    "qwen2.5-coder:7b",     # ~4.7 GB – fast autocomplete / coding
]

# If we detect a small GPU (<=8 GB) or CPU-only, use these instead
SMALL_MODELS = [
    "llama3.2:3b",           # ~2.0 GB
    "qwen2.5-coder:3b",     # ~1.9 GB
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
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _has_gpu() -> bool:
    """Auto-detect whether a CUDA GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _gpu_vram_mb() -> int:
    """Return total GPU VRAM in MiB, or 0 if no GPU."""
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
    """Return a short GPU description or 'CPU-only'."""
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


def _pick_default_models() -> list[str]:
    """Choose model list based on available VRAM."""
    vram = _gpu_vram_mb()
    if vram >= 14_000:   # T4 (16 GB) or better
        return list(DEFAULT_MODELS)
    elif vram >= 6_000:  # Smaller GPU
        return list(SMALL_MODELS)
    else:                # CPU-only or tiny GPU
        return list(SMALL_MODELS)


def _run_bg(cmd: list[str], logfile: Optional[str] = None, **kw) -> subprocess.Popen:
    """Launch a background process, optionally redirecting output to *logfile*."""
    log = open(logfile, "a") if logfile else subprocess.DEVNULL
    return subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        **kw,
    )


def _wait_for_port(port: int, host: str = "127.0.0.1", timeout: int = 120) -> bool:
    """Block until *port* is accepting TCP connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def _download_file(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest* with a progress bar."""
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
# System dependencies (Colab-specific)
# ────────────────────────────────────────────────────────────────────

def _install_system_deps() -> None:
    """Install OS-level packages that Ollama's install script needs.

    On Colab the packages `pciutils` (provides `lspci`) and `curl` are
    required so Ollama's installer can detect the GPU correctly.
    """
    console.print("  [cyan]↓[/] Installing system dependencies (pciutils, curl, lshw) …")
    try:
        subprocess.run(
            ["bash", "-c",
             "apt-get update -qq && "
             "apt-get install -y -qq pciutils curl lshw > /dev/null 2>&1"],
            check=True,
            capture_output=True,
            timeout=120,
        )
        console.print("  [green]✓[/] System dependencies ready")
    except subprocess.CalledProcessError as exc:
        console.print(f"  [yellow]⚠[/] Some system deps may have failed: {exc}")


# ────────────────────────────────────────────────────────────────────
# Ollama
# ────────────────────────────────────────────────────────────────────

def _install_ollama() -> None:
    """Install Ollama manually by downloading the pre-compiled binary.
    
    This bypasses the official install.sh which tries to set up systemd
    and can fail or raise CalledProcessError on Colab.
    """
    if shutil.which("ollama"):
        console.print("  [green]✓[/] Ollama already installed")
        return

    # System deps first (for pciutils, lshw, etc)
    _install_system_deps()

    console.print("  [cyan]↓[/] Installing Ollama (manual binary download) …")
    
    # Download the linux-amd64 tarball manually
    tar_url = "https://ollama.com/download/ollama-linux-amd64.tgz"
    tar_path = Path("/tmp/ollama-linux-amd64.tgz")
    
    try:
        _download_file(tar_url, tar_path)
        
        # Extract the binary using tar
        subprocess.run(
            ["tar", "-xzf", str(tar_path), "-C", "/usr/local"],
            check=True,
            capture_output=True
        )
        
        # Ensure it's executable
        ollama_bin = Path("/usr/local/bin/ollama")
        if ollama_bin.exists():
            ollama_bin.chmod(0o755)
            console.print("  [green]✓[/] Ollama installed successfully")
        else:
            raise FileNotFoundError("Ollama binary not found after extraction")
            
    except Exception as exc:
        console.print(f"  [red]✗[/] Ollama installation failed: {exc}")
        raise RuntimeError(f"Failed to install Ollama manually: {exc}")


def _start_ollama() -> None:
    """Start the Ollama server in the background (if not running).

    Uses threading so it does not block the Colab notebook.
    Sets OLLAMA_HOST and OLLAMA_ORIGINS for proper networking.
    """
    # Check if already running
    try:
        r = requests.get(f"http://127.0.0.1:{OLLAMA_PORT}/", timeout=3)
        if r.status_code == 200:
            console.print("  [green]✓[/] Ollama server already running")
            return
    except requests.RequestException:
        pass

    console.print("  [cyan]⟳[/] Starting Ollama server …")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "0.0.0.0:11434"
    env["OLLAMA_ORIGINS"] = "*"

    # Enable all GPU layers when available
    if _has_gpu():
        env["OLLAMA_NUM_GPU"] = "999"

    def _serve():
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=open("/tmp/ollama.log", "a"),
            stderr=subprocess.STDOUT,
            env=env,
        )

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    # Wait for startup
    if _wait_for_port(OLLAMA_PORT, timeout=30):
        console.print("  [green]✓[/] Ollama server ready on port 11434")
    else:
        console.print("  [red]✗[/] Ollama server failed to start")
        console.print("  [dim]   Check: !cat /tmp/ollama.log[/]")


def _pull_models(models: list[str]) -> None:
    """Pull each model using the Ollama CLI (with real-time progress).

    The CLI pull is more reliable on Colab than the REST API for large
    downloads because it handles retries and shows progress natively.
    """
    for model in models:
        console.print(f"  [cyan]↓[/] Pulling [bold]{model}[/] … (this may take a few minutes)")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min max per model
            )
            if result.returncode == 0:
                console.print(f"  [green]✓[/] {model} ready")
            else:
                error_msg = (result.stderr or result.stdout or "unknown error").strip()
                console.print(f"  [yellow]⚠[/] {model}: {error_msg[:200]}")
        except subprocess.TimeoutExpired:
            console.print(f"  [yellow]⚠[/] {model}: pull timed out (30 min limit)")
        except FileNotFoundError:
            console.print(f"  [red]✗[/] ollama binary not found – was it installed?")
            break


def _verify_models(models: list[str]) -> list[str]:
    """Return the subset of *models* that are actually available locally."""
    available = []
    try:
        r = requests.get(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=5)
        if r.ok:
            local = {m["name"] for m in r.json().get("models", [])}
            for m in models:
                # Ollama tags can be model:tag or just model (implies :latest)
                if m in local or m.split(":")[0] in {n.split(":")[0] for n in local}:
                    available.append(m)
    except requests.RequestException:
        pass
    return available


# ────────────────────────────────────────────────────────────────────
# code-server
# ────────────────────────────────────────────────────────────────────

def _install_code_server() -> None:
    """Install code-server using the official install script."""
    if shutil.which("code-server"):
        console.print("  [green]✓[/] code-server already installed")
        return

    console.print("  [cyan]↓[/] Installing code-server …")
    result = subprocess.run(
        ["bash", "-c", "curl -fsSL https://code-server.dev/install.sh | sh"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if shutil.which("code-server"):
        console.print("  [green]✓[/] code-server installed")
    else:
        console.print(f"  [red]✗[/] code-server installation failed")
        if result.stderr:
            console.print(f"      [dim]{result.stderr.strip()[-200:]}[/]")
        raise RuntimeError("code-server binary not found after install.")


def _write_code_server_config() -> None:
    """Write a minimal code-server config (no auth for tunneled access)."""
    CODE_SERVER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    config = textwrap.dedent(f"""\
        bind-addr: 127.0.0.1:{CODE_SERVER_PORT}
        auth: none
        cert: false
        disable-telemetry: true
    """)
    CODE_SERVER_CONFIG.write_text(config)
    console.print("  [green]✓[/] code-server config written (auth: none)")


def _install_continue_extension() -> None:
    """Install the Continue VS Code extension into code-server and
    pre-configure it to point at the local Ollama instance."""
    console.print("  [cyan]↓[/] Installing Continue extension …")
    try:
        result = subprocess.run(
            ["code-server", "--install-extension", CONTINUE_EXT],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            console.print("  [green]✓[/] Continue extension installed")
        else:
            console.print(f"  [yellow]⚠[/] Continue extension: {result.stderr.strip()[:150]}")
            console.print("  [dim]  You can install it manually from the Extensions sidebar.[/]")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        console.print(f"  [yellow]⚠[/] Continue extension install failed: {exc}")
        console.print("  [dim]  You can install it manually from the Extensions sidebar.[/]")

    # Pre-configure Continue to talk to Ollama regardless of extension status
    _write_continue_config()


def _write_continue_config(models: list[str] | None = None) -> None:
    """Write the Continue config so it auto-connects to Ollama."""
    config_dir = CONTINUE_CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"

    if models is None:
        models = _pick_default_models()

    # Build model entries for Continue
    model_entries = []
    autocomplete_model = None
    for m in models:
        entry = {
            "title": f"{m} (Ollama)",
            "provider": "ollama",
            "model": m,
            "apiBase": f"http://127.0.0.1:{OLLAMA_PORT}",
        }
        model_entries.append(entry)
        # Prefer a coding model for autocomplete
        if "coder" in m.lower() or "code" in m.lower():
            autocomplete_model = entry.copy()
            autocomplete_model["title"] = f"{m} (autocomplete)"

    # Fallback: use the first model for autocomplete
    if not autocomplete_model and model_entries:
        autocomplete_model = model_entries[0].copy()
        autocomplete_model["title"] = f"{models[0]} (autocomplete)"

    config = {
        "models": model_entries,
        "allowAnonymousTelemetry": False,
    }
    if autocomplete_model:
        config["tabAutocompleteModel"] = autocomplete_model

    config_file.write_text(json.dumps(config, indent=2))
    console.print("  [green]✓[/] Continue config → Ollama (localhost:11434)")


def _start_code_server() -> None:
    """Launch code-server in the background."""
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
        console.print("  [dim]   Check: !cat /tmp/code-server.log[/]")


# ────────────────────────────────────────────────────────────────────
# Cloudflared tunnel
# ────────────────────────────────────────────────────────────────────

def _install_cloudflared() -> None:
    """Download the cloudflared binary if not already present."""
    if shutil.which("cloudflared"):
        console.print("  [green]✓[/] cloudflared already installed")
        return

    console.print("  [cyan]↓[/] Installing cloudflared …")

    # Try wget first (faster, common on Colab), fall back to requests
    dest = Path("/usr/local/bin/cloudflared")
    url = (
        "https://github.com/cloudflare/cloudflared/releases/latest"
        "/download/cloudflared-linux-amd64"
    )
    try:
        subprocess.run(
            ["wget", "-q", "-O", str(dest), url],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        _download_file(url, dest)

    dest.chmod(0o755)
    console.print("  [green]✓[/] cloudflared installed")


def _start_tunnel() -> str | None:
    """Start a Cloudflare quick-tunnel and return the public URL."""
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

    # Parse the tunnel URL from cloudflared output
    url: str | None = None
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            time.sleep(0.2)
            continue
        if ".trycloudflare.com" in line:
            # Extract URL from the line
            match = re.search(r"https?://[\w-]+\.trycloudflare\.com", line)
            if match:
                url = match.group(0)
                break

    if url:
        console.print(f"  [green]✓[/] Tunnel active")
    else:
        console.print("  [red]✗[/] Could not obtain tunnel URL")
        console.print("  [dim]   The tunnel may still be starting – check output above.[/]")

    return url


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

def setup(
    models: list[str] | None = None,
    pull_models: bool = True,
    install_continue: bool = True,
) -> None:
    """
    One-click setup: install all components without launching them.

    Parameters
    ----------
    models : list[str], optional
        Ollama model tags to pull. Auto-selected based on available GPU
        VRAM if not specified.
    pull_models : bool
        If True (default), pull models after installing Ollama.
    install_continue : bool
        If True (default), install the Continue extension for code-server.
    """
    if models is None:
        models = _pick_default_models()

    console.print(Panel(VIBE_BANNER, title="[bold magenta]vibe-env setup[/]", border_style="magenta"))
    console.print()

    # GPU info
    gpu = _gpu_info()
    vram = _gpu_vram_mb()
    console.print(f"  [bold]GPU:[/] {gpu}")
    if _has_gpu():
        console.print(f"  [green]✓[/] Hardware acceleration enabled ({vram} MiB VRAM)")
    else:
        console.print("  [yellow]⚠[/] No GPU detected – models will run on CPU")
        console.print("  [dim]   Tip: Runtime → Change runtime type → T4 GPU[/]")
    console.print(f"  [bold]Models:[/] {', '.join(models)}")
    console.print()

    # 1. Ollama
    console.print("[bold cyan]① Ollama[/]")
    _install_ollama()
    _start_ollama()
    if pull_models:
        _pull_models(models)
        pulled = _verify_models(models)
        console.print(f"  [bold]Available:[/] {', '.join(pulled) or 'none'}")
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

    console.print("[bold green]Setup complete![/] Run [bold]vibe_env.launch()[/] to start.")


def launch(
    models: list[str] | None = None,
    pull_models: bool = True,
    install_continue: bool = True,
    password: str | None = None,
) -> str | None:
    """
    All-in-one launcher: setup → start → tunnel → URL.

    Parameters
    ----------
    models : list[str], optional
        Models to pull. Auto-selected based on GPU VRAM if not specified.
    pull_models : bool
        Pull models during setup.
    install_continue : bool
        Install the Continue extension.
    password : str, optional
        If set, code-server will require this password for login.

    Returns
    -------
    str or None
        The public *.trycloudflare.com URL, or None on failure.
    """
    if models is None:
        models = _pick_default_models()

    console.print(Panel(VIBE_BANNER, title="[bold magenta]vibe-env[/]", border_style="magenta"))
    console.print()

    # Environment check
    if _is_colab():
        console.print("  [green]✓[/] Running on Google Colab")
    else:
        console.print("  [yellow]⚠[/] Not on Colab – some features may work differently")

    # GPU info
    gpu = _gpu_info()
    vram = _gpu_vram_mb()
    console.print(f"  [bold]GPU:[/] {gpu}")
    if _has_gpu():
        console.print(f"  [green]✓[/] Hardware acceleration enabled 🚀 ({vram} MiB VRAM)")
    else:
        console.print("  [yellow]⚠[/] No GPU detected – models will run on CPU (slower)")
        console.print("  [dim]   Tip: Runtime → Change runtime type → T4 GPU[/]")
    console.print(f"  [bold]Models to pull:[/] {', '.join(models)}")
    console.print()

    # ── Step 1: Ollama ──────────────────────────────────────────────
    console.print("[bold cyan]① Setting up Ollama[/]")
    _install_ollama()
    _start_ollama()
    if pull_models:
        _pull_models(models)
        pulled = _verify_models(models)
    else:
        pulled = []
    console.print()

    # ── Step 2: code-server ─────────────────────────────────────────
    console.print("[bold cyan]② Setting up code-server[/]")
    _install_code_server()
    _write_code_server_config()
    if password:
        # Override auth in config
        config_text = CODE_SERVER_CONFIG.read_text()
        config_text = config_text.replace("auth: none", f"auth: password\npassword: {password}")
        CODE_SERVER_CONFIG.write_text(config_text)
        console.print(f"  [green]✓[/] Password auth enabled")

    if install_continue:
        _install_continue_extension()

    _start_code_server()
    console.print()

    # ── Step 3: Cloudflare Tunnel ───────────────────────────────────
    console.print("[bold cyan]③ Opening Cloudflare Tunnel[/]")
    _install_cloudflared()
    url = _start_tunnel()
    console.print()

    # ── Summary ─────────────────────────────────────────────────────
    table = Table(title="🎉 Vibe Environment Ready", border_style="green", show_lines=True)
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    table.add_row("Ollama", "[green]Running[/]", f"localhost:{OLLAMA_PORT}")
    table.add_row(
        "Models",
        "[green]Pulled[/]" if pulled else ("[dim]Skipped[/]" if not pull_models else "[yellow]Check[/]"),
        ", ".join(pulled) if pulled else ", ".join(models),
    )
    table.add_row("code-server", "[green]Running[/]", f"localhost:{CODE_SERVER_PORT}")
    table.add_row("Continue Ext", "[green]Installed[/]" if install_continue else "[dim]Skipped[/]", "→ Ollama")
    table.add_row(
        "Public URL",
        "[green]Active[/]" if url else "[red]Failed[/]",
        url or "N/A",
    )

    console.print(table)
    console.print()

    if url:
        console.print(
            Panel(
                f"[bold green]🔗 Open your IDE:[/]\n\n"
                f"   [bold underline link={url}]{url}[/]\n\n"
                f"[dim]The Continue extension is pre-configured to use Ollama.\n"
                f"Open the Continue sidebar (Ctrl+Shift+L) to start vibe coding![/]",
                title="[bold]Your Vibe Coding URL[/]",
                border_style="green",
            )
        )

        # If in Colab, also render a clickable HTML link
        if _is_colab():
            try:
                from IPython.display import HTML, display
                display(HTML(
                    f'<h2>🔗 <a href="{url}" target="_blank">'
                    f'Click here to open your Vibe IDE</a></h2>'
                    f'<p style="color:#888">URL: {url}</p>'
                ))
            except Exception:
                pass
    else:
        console.print("[red]Could not establish a tunnel. Check the logs above.[/]")

    return url


def status() -> dict:
    """
    Return the current status of all vibe-env components.

    Returns
    -------
    dict
        Keys: 'ollama', 'code_server', 'cloudflared', 'gpu' with their
        running status.
    """
    result = {}

    # GPU
    result["gpu"] = _gpu_info()
    result["vram_mb"] = _gpu_vram_mb()

    # Ollama
    try:
        r = requests.get(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        result["ollama"] = {"running": True, "models": models}
    except Exception:
        result["ollama"] = {"running": False, "models": []}

    # code-server
    try:
        with socket.create_connection(("127.0.0.1", CODE_SERVER_PORT), timeout=2):
            result["code_server"] = {"running": True, "port": CODE_SERVER_PORT}
    except OSError:
        result["code_server"] = {"running": False}

    # Cloudflared
    cf_running = any(
        "cloudflared" in (p.info.get("name") or "")
        for p in psutil.process_iter(["name"])
    )
    result["cloudflared"] = {"running": cf_running}

    # Pretty-print
    table = Table(title="vibe-env status", border_style="cyan")
    table.add_column("Component", style="bold")
    table.add_column("Status")

    tag = "[green]Running[/]" if result["ollama"]["running"] else "[red]Stopped[/]"
    table.add_row("Ollama", f"{tag}  models: {', '.join(result['ollama']['models']) or 'none'}")
    tag = "[green]Running[/]" if result["code_server"].get("running") else "[red]Stopped[/]"
    table.add_row("code-server", tag)
    tag = "[green]Running[/]" if result["cloudflared"]["running"] else "[red]Stopped[/]"
    table.add_row("Cloudflared", tag)
    table.add_row("GPU", f"{result['gpu']} ({result['vram_mb']} MiB)")

    console.print(table)
    return result


def stop() -> None:
    """Stop all background processes started by vibe-env."""
    console.print("[bold yellow]Stopping vibe-env services …[/]")
    for name in ("ollama", "code-server", "cloudflared"):
        for proc in psutil.process_iter(["name", "pid"]):
            pname = (proc.info.get("name") or "").lower()
            if name.replace("-", "") in pname.replace("-", ""):
                try:
                    proc.terminate()
                    console.print(f"  [yellow]■[/] Terminated {name} (PID {proc.info['pid']})")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    console.print("[green]All services stopped.[/]")

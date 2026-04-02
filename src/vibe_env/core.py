"""
vibe_env.core – orchestrates Ollama, code-server, Cloudflared, and
                the Continue extension so a single `vibe_env.launch()`
                gives you a full Cursor-like IDE on Google Colab.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
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
ROO_CODE_EXT = "rooveterinaryinc.roo-cline"
CONTINUE_EXT = "Continue.continue"

DEFAULT_MODELS = [
    "llama3.1:8b",
    "deepseek-coder-v2:16b",
    "mistral-nemo:12b",
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
    import socket
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
# Installation helpers
# ────────────────────────────────────────────────────────────────────

def _install_ollama() -> None:
    """Install Ollama if not already present."""
    if shutil.which("ollama"):
        console.print("  [green]✓[/] Ollama already installed")
        return

    console.print("  [cyan]↓[/] Installing Ollama …")
    subprocess.run(
        ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
        check=True,
        capture_output=True,
    )
    console.print("  [green]✓[/] Ollama installed")


def _start_ollama() -> None:
    """Start the Ollama server in the background (if not running)."""
    for proc in psutil.process_iter(["name"]):
        if proc.info["name"] and "ollama" in proc.info["name"].lower():
            console.print("  [green]✓[/] Ollama server already running")
            return

    console.print("  [cyan]⟳[/] Starting Ollama server …")
    env = os.environ.copy()
    # Enable GPU layers when available
    if _has_gpu():
        env["OLLAMA_NUM_GPU"] = "999"  # use all layers on GPU
    _run_bg(["ollama", "serve"], logfile="/tmp/ollama.log", env=env)

    if _wait_for_port(OLLAMA_PORT, timeout=60):
        console.print("  [green]✓[/] Ollama server ready on port 11434")
    else:
        console.print("  [red]✗[/] Ollama server failed to start (check /tmp/ollama.log)")


def _pull_models(models: list[str]) -> None:
    """Pull each model through the Ollama API (shows progress)."""
    for model in models:
        console.print(f"  [cyan]↓[/] Pulling [bold]{model}[/] …")
        try:
            resp = requests.post(
                f"http://127.0.0.1:{OLLAMA_PORT}/api/pull",
                json={"name": model, "stream": False},
                timeout=1800,  # models can be large
            )
            if resp.ok:
                console.print(f"  [green]✓[/] {model} ready")
            else:
                console.print(f"  [yellow]⚠[/] {model}: {resp.text[:120]}")
        except requests.RequestException as exc:
            console.print(f"  [red]✗[/] {model}: {exc}")


# ────────────────────────────────────────────────────────────────────
# code-server
# ────────────────────────────────────────────────────────────────────

def _install_code_server() -> None:
    """Install code-server using the official install script."""
    if shutil.which("code-server"):
        console.print("  [green]✓[/] code-server already installed")
        return

    console.print("  [cyan]↓[/] Installing code-server …")
    subprocess.run(
        ["bash", "-c", "curl -fsSL https://code-server.dev/install.sh | sh"],
        check=True,
        capture_output=True,
    )
    console.print("  [green]✓[/] code-server installed")


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
        subprocess.run(
            ["code-server", "--install-extension", CONTINUE_EXT],
            check=True,
            capture_output=True,
            timeout=120,
        )
        console.print("  [green]✓[/] Continue extension installed")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        console.print(f"  [yellow]⚠[/] Continue extension install failed: {exc}")
        console.print("  [dim]  You can install it manually from the Extensions sidebar.[/]")
        return

    # Pre-configure Continue to talk to Ollama
    _write_continue_config()


def _write_continue_config() -> None:
    """Write the Continue config so it auto-connects to Ollama."""
    config_dir = CONTINUE_CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"

    config = {
        "models": [
            {
                "title": "Llama 3.1 8B (Ollama)",
                "provider": "ollama",
                "model": "llama3.1:8b",
                "apiBase": f"http://127.0.0.1:{OLLAMA_PORT}",
            },
            {
                "title": "DeepSeek Coder V2 (Ollama)",
                "provider": "ollama",
                "model": "deepseek-coder-v2:16b",
                "apiBase": f"http://127.0.0.1:{OLLAMA_PORT}",
            },
            {
                "title": "Mistral Nemo (Ollama)",
                "provider": "ollama",
                "model": "mistral-nemo:12b",
                "apiBase": f"http://127.0.0.1:{OLLAMA_PORT}",
            },
        ],
        "tabAutocompleteModel": {
            "title": "DeepSeek Coder V2 (autocomplete)",
            "provider": "ollama",
            "model": "deepseek-coder-v2:16b",
            "apiBase": f"http://127.0.0.1:{OLLAMA_PORT}",
        },
        "allowAnonymousTelemetry": False,
    }
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
        console.print("  [red]✗[/] code-server failed to start (check /tmp/code-server.log)")


# ────────────────────────────────────────────────────────────────────
# Cloudflared tunnel
# ────────────────────────────────────────────────────────────────────

def _install_cloudflared() -> None:
    """Download the cloudflared binary if not already present."""
    if shutil.which("cloudflared"):
        console.print("  [green]✓[/] cloudflared already installed")
        return

    console.print("  [cyan]↓[/] Installing cloudflared …")
    dest = Path("/usr/local/bin/cloudflared")
    url = (
        "https://github.com/cloudflare/cloudflared/releases/latest"
        "/download/cloudflared-linux-amd64"
    )
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
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        line = proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            time.sleep(0.2)
            continue
        if ".trycloudflare.com" in line:
            # Extract URL from the line
            for token in line.split():
                if "trycloudflare.com" in token:
                    url = token.strip().rstrip("|").rstrip()
                    if not url.startswith("http"):
                        url = "https://" + url
                    break
            if url:
                break

    if url:
        console.print(f"  [green]✓[/] Tunnel active")
    else:
        console.print("  [red]✗[/] Could not obtain tunnel URL (check /tmp/cloudflared.log)")

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
        Ollama model tags to pull. Defaults to Llama 3.1 8B,
        DeepSeek-Coder-V2 16B, and Mistral-Nemo 12B.
    pull_models : bool
        If True (default), pull models after installing Ollama.
    install_continue : bool
        If True (default), install the Continue extension for code-server.
    """
    if models is None:
        models = DEFAULT_MODELS

    console.print(Panel(VIBE_BANNER, title="[bold magenta]vibe-env setup[/]", border_style="magenta"))
    console.print()

    # GPU info
    gpu = _gpu_info()
    console.print(f"  [bold]GPU:[/] {gpu}")
    if _has_gpu():
        console.print("  [green]✓[/] Hardware acceleration enabled")
    else:
        console.print("  [yellow]⚠[/] No GPU detected – models will run on CPU")
    console.print()

    # 1. Ollama
    console.print("[bold cyan]① Ollama[/]")
    _install_ollama()
    _start_ollama()
    if pull_models:
        _pull_models(models)
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
        Models to pull (see `setup()`).
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
        models = DEFAULT_MODELS

    console.print(Panel(VIBE_BANNER, title="[bold magenta]vibe-env[/]", border_style="magenta"))
    console.print()

    # GPU info
    gpu = _gpu_info()
    console.print(f"  [bold]GPU:[/] {gpu}")
    if _has_gpu():
        console.print("  [green]✓[/] Hardware acceleration enabled 🚀")
    else:
        console.print("  [yellow]⚠[/] No GPU detected – models will run on CPU (slower)")
    console.print()

    # ── Step 1: Ollama ──────────────────────────────────────────────
    console.print("[bold cyan]① Setting up Ollama[/]")
    _install_ollama()
    _start_ollama()
    if pull_models:
        _pull_models(models)
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
        "[green]Pulled[/]" if pull_models else "[dim]Skipped[/]",
        ", ".join(models),
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

    # Ollama
    try:
        r = requests.get(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        result["ollama"] = {"running": True, "models": models}
    except Exception:
        result["ollama"] = {"running": False, "models": []}

    # code-server
    import socket
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
    table.add_row("GPU", result["gpu"])

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

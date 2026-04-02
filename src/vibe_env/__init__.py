"""
vibe-env: Turn Google Colab into a Vibe Coding powerhouse.

Usage in a Colab cell:
    !pip install git+https://github.com/your-username/vibe-colab.git
    import vibe_env
    vibe_env.launch()
"""

__version__ = "0.1.0"

from vibe_env.core import launch, setup, status, stop

__all__ = ["launch", "setup", "status", "stop"]

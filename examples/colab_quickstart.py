"""
Vibe Env — Google Colab Quick Start
====================================

Copy these cells into a new Google Colab notebook.
Make sure to select a GPU runtime (T4 or better recommended).

Runtime → Change runtime type → GPU
"""

# ── Cell 1: Install ────────────────────────────────────────────────
# !pip install git+https://github.com/MeowMan007/vibe-colab.git

# ── Cell 2: Launch (default — pulls 3 models) ─────────────────────
# import vibe_env
# url = vibe_env.launch()
# # Click the printed URL to open your IDE!

# ── Cell 2 (alt): Launch with custom models ────────────────────────
# import vibe_env
# url = vibe_env.launch(
#     models=["codellama:7b", "phi3:mini"],
#     password="my-secret",  # optional: protect your IDE
# )

# ── Cell 3 (optional): Check status ───────────────────────────────
# vibe_env.status()

# ── Cell 4 (optional): Stop everything ─────────────────────────────
# vibe_env.stop()

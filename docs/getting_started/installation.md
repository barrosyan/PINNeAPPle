# Installation

```bash
git clone https://github.com/barrosyan/PINNeAPPle.git
cd PINNeAPPle
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -U pip
pip install -e ".[dev]"
```

Verify:

```bash
python -c "import pinneaple_arena, pinneaple_pinn; print('OK')"
```

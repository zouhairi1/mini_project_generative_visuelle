"""Copy the project code from the laptop into a notebook, so a Colab kernel can use it without git.

With the VS Code Colab extension, the notebook file is on the laptop but the code runs on Google's
machine, which cannot see the laptop files. This script zips src/*.py, configs/*.yaml and
data/download.py, and writes them (base64) into the `CODE_ZIP = "..."` line of the notebook's setup cell.
On Colab, the setup cell unzips them and uses them.

Run it again after every change in src/ or configs/ (from the repository root):
    python -m src.sync_code notebooks/03a_cvae.ipynb notebooks/03b_cgan.ipynb
"""
import base64
import io
import json
import sys
import zipfile
from pathlib import Path

from .utils import ROOT

MARKER = "CODE_ZIP = "


def code_zip_b64() -> str:
    files = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "configs").glob("*.yaml")) + [ROOT / "data" / "download.py"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    return base64.b64encode(buf.getvalue()).decode()


def sync(nb_path) -> None:
    nb_path = Path(nb_path)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    payload = code_zip_b64()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        lines = src.split("\n")
        hits = [i for i, line in enumerate(lines) if line.startswith(MARKER)]
        if hits:
            lines[hits[0]] = f'{MARKER}"{payload}"'
            cell["source"] = "\n".join(lines)
            nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"{nb_path.name}: code copied into the setup cell ({len(payload) / 1000:.0f} kB)")
            return
    print(f"{nb_path.name}: no line starting with {MARKER!r} found, nothing changed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        sync(path)

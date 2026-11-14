"""Recover result files printed by a Colab kernel into a local notebook (VS Code + Colab).

With the VS Code Colab extension the code runs on Google's machine, so files it writes
stay there, and `files.download()` does not work. Instead, the last notebook cell prints
the results zip as base64 text between ZIP_BASE64_START / ZIP_BASE64_END markers. VS Code
saves that output in the local .ipynb; this script decodes it, extracts the files into
the repository, and replaces the long text output with a short note.

Usage (from the repository root, after saving the notebook in VS Code):
    python -m src.extract_results notebooks/02_baseline.ipynb
"""
import base64
import io
import json
import sys
import zipfile
from pathlib import Path

from .utils import ROOT

START, END = "ZIP_BASE64_START", "ZIP_BASE64_END"


def _text(out) -> str:
    t = out.get("text", "")
    return "".join(t) if isinstance(t, list) else t


def extract(nb_path) -> list:
    nb_path = Path(nb_path)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    extracted = []
    for cell in nb["cells"]:
        outputs = cell.get("outputs", [])
        stream = "".join(_text(o) for o in outputs if o.get("output_type") == "stream")
        if START not in stream or END not in stream:
            continue
        body = stream.split(START, 1)[1].split(END, 1)[0]
        payload = "".join(body.splitlines()[1:])          # first line = header (name, size)
        data = base64.b64decode(payload)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(ROOT)
            extracted += z.namelist()
        note = f"[{len(data)} bytes extracted locally with src.extract_results: {len(extracted)} files]\n"
        cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": note}]

    if not extracted:
        print("No ZIP_BASE64 output found. Run the export cell on Colab, save the notebook (Ctrl+S), then retry.")
        return []

    # Drop JavaScript outputs (from files.download) that only clutter the notebook.
    for cell in nb["cells"]:
        if "outputs" in cell:
            cell["outputs"] = [o for o in cell["outputs"] if "application/javascript" not in o.get("data", {})]

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Extracted into", ROOT)
    for name in extracted:
        print("  ", name)
    return extracted


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    extract(sys.argv[1])

"""Download pinned upstream code and official, unquantized model weights."""

import ast
import hashlib
import io
import os
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
DOVER_REV = "f1ddc96215bc7fbcf8f315c65d47905f339c3419"
DOVER_WEIGHTS_REV = "841d2f90f8ed5e30e7d60213fd6b4354fd56b06c"
QWEN_REV = "89644892e4d85e24eaac8bacfd4f463576704203"
HASHES = {
    "dover/DOVER.pth": "f4a42c0bbc94c94dd7409e7f40887d44c5c30314d1d09e7edf03cc35813b4838",
    "qwen/model.safetensors": "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
}


def prepare_code():
    target = ROOT / ".vendor" / "DOVER"
    url = f"https://codeload.github.com/VQAssessment/DOVER/zip/{DOVER_REV}"
    with urllib.request.urlopen(url) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for entry in archive.infolist():
        parts = Path(entry.filename).parts[1:]
        if not parts or ".." in parts or entry.is_dir():
            continue
        relative = "/".join(parts)
        if relative.startswith("dover/models/") or relative in {
            "dover/datasets/dover_datasets.py",
            "dover.yml",
            "LICENSE",
            "LICENSE.txt",
        }:
            path = target.joinpath(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(entry))
    (target / "REVISION").write_text(DOVER_REV + "\n")
    # Avoid importing training datasets and their unused Decord dependency.
    (target / "dover/__init__.py").write_text("from .models import *\n")
    evaluator = target / "dover/models/evaluator.py"
    source = evaluator.read_text()
    assert "convnext_3d_tiny(pretrained=True)" in source
    # A complete DOVER state_dict is loaded strictly; ImageNet initialization is redundant.
    evaluator.write_text(source.replace("convnext_3d_tiny(pretrained=True)", "convnext_3d_tiny(pretrained=False)"))
    source = (target / "dover/datasets/dover_datasets.py").read_text()
    names = {
        "get_spatial_fragments",
        "get_resize_function",
        "get_resized_video",
        "get_single_view",
        "UnifiedFrameSampler",
    }
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    imports = "import random\nfrom functools import lru_cache\nimport numpy as np\nimport torch\nimport torchvision\n"
    (ROOT / ".vendor/dover_views.py").write_text(
        "# Generated verbatim in semantics from pinned DOVER; see prepare.py.\n"
        + imports
        + "\n\n"
        + "\n\n".join(ast.unparse(n) for n in nodes)
        + "\n"
    )
    print("DOVER source prepared:", DOVER_REV, flush=True)


def main():
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download

    prepare_code()
    for repo, revision, folder, patterns in [
        ("teowu/DOVER", DOVER_WEIGHTS_REV, "dover", ["DOVER.pth", "README.md", "LICENSE*"]),
        (
            "Qwen/Qwen3-VL-2B-Instruct",
            QWEN_REV,
            "qwen",
            ["*.json", "*.safetensors", "merges.txt", "vocab.json", "README.md", "LICENSE*"],
        ),
    ]:
        print("Downloading", repo, flush=True)
        snapshot_download(
            repo,
            revision=revision,
            local_dir=ROOT / ".models" / folder,
            allow_patterns=patterns,
            token=False,
            max_workers=3,
        )
    for filename, expected in HASHES.items():
        with (ROOT / ".models" / filename).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {filename}")
        print("Verified", filename, flush=True)


if __name__ == "__main__":
    main()

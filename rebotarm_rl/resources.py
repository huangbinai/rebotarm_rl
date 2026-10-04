"""Fetch immutable model resources; no ROS checkout or runtime dependency."""
import argparse
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


def manifest():
    return json.loads(files("rebotarm_rl").joinpath("model_manifest.json").read_text())


def model_directory():
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "rebotarm_rl" / "models" / manifest()["commit"]


def validate_model(directory):
    for name, expected in manifest()["files"].items():
        path = Path(directory) / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or changed pinned model resource: {path}")


def fetch_model(source=None):
    """Install verified bytes atomically per file, resuming interrupted downloads.

    source optionally supplies an existing model directory, but must match the
    manifest byte for byte. It is copied into the independent cache, never linked.
    """
    spec = manifest()
    directory = model_directory()
    for name, expected in spec["files"].items():
        target = directory / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        if source is not None:
            content = (Path(source) / name).read_bytes()
        else:
            repo = spec["repository"].removeprefix("https://github.com/")
            url = f"https://raw.githubusercontent.com/{repo}/{spec['commit']}/{spec['root']}/{name}"
            with urlopen(url, timeout=60) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Model checksum mismatch: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temp = Path(stream.name)
            stream.write(content)
        try:
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    validate_model(directory)
    return directory / "reach_scene.xml"


def reach_scene_path():
    configured = os.environ.get("REBOTARM_MJLAB_SCENE")
    if configured:
        # Explicit custom models are allowed for experiments; they are not
        # claimed to match the pinned baseline.
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configured scene does not exist: {path}")
        return path
    directory = model_directory()
    try:
        validate_model(directory)
    except ValueError as exc:
        raise FileNotFoundError(
            "Pinned model unavailable or modified. Run rebotarm-rl-fetch-model, "
            "or explicitly set REBOTARM_MJLAB_SCENE for a custom model."
        ) from exc
    return directory / "reach_scene.xml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional matching local model directory")
    args = parser.parse_args()
    print(fetch_model(args.source))


if __name__ == "__main__":
    main()

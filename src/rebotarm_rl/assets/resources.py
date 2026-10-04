"""定位并校验随包分发的模型资源，无网络或外部缓存依赖。"""
import argparse
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path


def manifest() -> dict:
    return json.loads(files("rebotarm_rl.assets").joinpath("model_manifest.json").read_text())


def model_directory() -> Path:
    return Path(str(files("rebotarm_rl.assets").joinpath("rebotarm")))


def validate_model(directory: Path) -> None:
    for name, expected in manifest()["files"].items():
        path = Path(directory) / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or changed pinned model resource: {path}")


def reach_scene_path() -> Path:
    configured = os.environ.get("REBOTARM_MJLAB_SCENE")
    if configured:
        # 自定义实验模型需要显式指定，不宣称与固定模型基线一致。
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configured scene does not exist: {path}")
        return path
    directory = model_directory()
    try:
        validate_model(directory)
    except ValueError as exc:
        raise FileNotFoundError(
            "Bundled model unavailable or modified. Restore the repository assets or reinstall the package, "
            "or explicitly set REBOTARM_MJLAB_SCENE for a custom model."
        ) from exc
    return directory / "reach_scene.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(reach_scene_path())


if __name__ == "__main__":
    main()

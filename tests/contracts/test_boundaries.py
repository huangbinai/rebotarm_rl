"""检查包依赖边界，避免训练代码重新耦合ROS或硬件。"""
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src/rebotarm_rl"


def test_public_import_does_not_initialize_frameworks():
    subprocess.run([sys.executable, "-I", "-c",
        "import rebotarm_rl; import rebotarm_rl.contracts.policy; import sys; "
        "assert not {'torch','mujoco','mjlab','rclpy'} & sys.modules.keys()"], check=True)


def test_dependency_direction():
    for path in SRC.rglob("*.py"):
        modules = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        assert not any(m.split(".")[0] in {"rclpy", "rebotarmcontroller",
            "rebotarm_simulation", "rebotarm_preview", "isaaclab"} for m in modules), path
        if path.is_relative_to(SRC / "contracts") or path.is_relative_to(SRC / "assets"):
            assert not any(m.split(".")[0] in {"torch", "mjlab", "mujoco"} for m in modules)


def test_installed_plugin_points_to_backend():
    from importlib.metadata import entry_points
    plugins = [p for p in entry_points(group="mjlab.tasks") if p.name == "rebotarm_reach"]
    assert len(plugins) == 1
    assert plugins[0].value == "rebotarm_rl.backends.mjlab.registration"

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "rebotarm_rl/tasks/reach.py"


def test_mjlab_plugin_is_optional_and_registered_as_a_task_entrypoint() -> None:
    setup = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"mjlab.tasks"' in setup
    assert 'rebotarm_reach = "rebotarm_rl.tasks.reach"' in setup
    assert PLUGIN.is_file()


def test_mjlab_plugin_has_no_ros_or_hardware_imports() -> None:
    tree = ast.parse(PLUGIN.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert not any(module.startswith(("rclpy", "rebotarmcontroller", "serial")) for module in modules)


def test_mjlab_runner_maps_actor_and_critic_to_public_observations() -> None:
    source = (ROOT / "rebotarm_rl/agents/reach_ppo.py").read_text(encoding="utf-8")
    assert 'obs_groups={"actor": ("actor",), "critic": ("actor",)}' in source
    assert 'logger="tensorboard"' in source


def test_rl_has_single_registration_owner_and_no_ros_dependencies():
    for path in (ROOT / "rebotarm_rl").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else [])
            assert not any(m.split(".")[0] in {"rclpy", "rebotarm_simulation", "rebotarmcontroller", "rebotarm_preview"} for m in modules), path

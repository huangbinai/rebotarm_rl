"""Provenance hooks only; optimization and checkpoint mechanics remain upstream."""
import json
from pathlib import Path
import torch
from mjlab.rl import MjlabOnPolicyRunner
from rebotarm_rl.assets.resources import manifest, reach_scene_path
from rebotarm_rl.contracts.artifacts import file_hash, write_run_record
from rebotarm_rl.contracts.policy import REACH_V1
from .validation import validate_config


class RecordedRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        validate_config(env.unwrapped.cfg)
        obs = env.get_observations()
        REACH_V1.validate_shapes(obs["actor"].shape[-1], env.num_actions)
        super().__init__(env, train_cfg, log_dir, device)
        if log_dir and getattr(self, "global_rank", 0) == 0:
            scene = reach_scene_path()
            # Compiled MJB captures custom model changes and compiler settings.
            import mujoco
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            compiled = path / "compiled_model.mjb"
            mujoco.mj_saveModel(env.unwrapped.sim.mj_model, str(compiled), None)
            write_run_record(path, repository=Path(__file__).resolve().parents[4],
                model={"baseline": manifest(), "scene": str(scene),
                       "scene_sha256": file_hash(scene),
                       "compiled_model_sha256": file_hash(compiled)},
                runtime={"backend": "mjlab", "device": str(device),
                         "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                         "seed": env.unwrapped.cfg.seed})
        self._record_directory = Path(log_dir) if log_dir else None

    def load(self, path, *args, **kwargs):
        record_path = Path(path).parent / "run_manifest.json"
        if record_path.is_file():
            REACH_V1.require_compatible(json.loads(record_path.read_text())["contract"])
        else:
            import warnings
            warnings.warn("Legacy checkpoint has no contract manifest; compatibility is unverified")
        result = super().load(path, *args, **kwargs)
        if self._record_directory:
            record_path = self._record_directory / "run_manifest.json"
            if record_path.is_file():
                record = json.loads(record_path.read_text())
                record["resume"] = {"path": str(path), "sha256": file_hash(Path(path))}
                record_path.write_text(json.dumps(record, indent=2) + "\n")
        return result

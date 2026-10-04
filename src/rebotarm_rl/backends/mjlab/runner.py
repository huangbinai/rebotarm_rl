"""扩展实验记录与契约校验；优化循环和权重保存仍由上游执行。"""
import json
from pathlib import Path
import torch
from mjlab.rl import MjlabOnPolicyRunner
from rebotarm_rl.assets.resources import manifest, reach_scene_path
from rebotarm_rl.contracts.artifacts import file_hash, write_run_record
from .validation import validate_config


class RecordedRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.contract = validate_config(env.unwrapped.cfg)
        obs = env.get_observations()
        self.contract.validate_shapes(obs["actor"].shape[-1], env.num_actions)
        super().__init__(env, train_cfg, log_dir, device)
        if log_dir and getattr(self, "global_rank", 0) == 0:
            scene = reach_scene_path()
            # 编译后的MJB记录实际模型修改及编译设置。
            import mujoco
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            compiled = path / "compiled_model.mjb"
            mujoco.mj_saveModel(env.unwrapped.sim.mj_model, str(compiled), None)
            write_run_record(path, repository=Path(__file__).resolve().parents[4],
                model={"baseline": manifest(), "scene": str(scene),
                       "scene_sha256": file_hash(scene),
                       "compiled_model_sha256": file_hash(compiled)},
                contract=self.contract,
                runtime={"backend": "mjlab", "device": str(device),
                         "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                         "seed": env.unwrapped.cfg.seed})
        self._record_directory = Path(log_dir) if log_dir else None

    def save(self, path: str, infos=None) -> None:
        """随权重保存契约；保留上游环境计数器与上传逻辑。"""
        super().save(path, {**(infos or {}), "policy_contract": self.contract.to_dict()})

    def load(self, path, *args, **kwargs):
        validate_checkpoint_contract(Path(path), self.contract)
        result = super().load(path, *args, **kwargs)
        if self._record_directory:
            record_path = self._record_directory / "run_manifest.json"
            if record_path.is_file():
                record = json.loads(record_path.read_text())
                record["resume"] = {"path": str(path), "sha256": file_hash(Path(path))}
                record_path.write_text(json.dumps(record, indent=2) + "\n")
        return result


def validate_checkpoint_contract(path, contract):
    """在加载任何网络或优化器状态前验证内嵌契约和旁置清单。

    仅对用户信任的checkpoint使用PyTorch加载。V1允许带警告加载旧无契约权重；
    V2必须有内嵌契约，避免仅复制权重后丢失版本身份。
    """
    from rebotarm_rl.contracts.policy import REACH_V2
    payload = torch.load(path, map_location="cpu", weights_only=False)
    embedded = (payload.get("infos") or {}).get("policy_contract")
    if embedded is not None:
        contract.require_compatible(embedded)
    elif contract.version == REACH_V2.version:
        raise ValueError("V2拒绝无内嵌契约的checkpoint；请从V2任务重新训练")
    record_path = Path(path).parent / "run_manifest.json"
    if record_path.is_file():
        contract.require_compatible(json.loads(record_path.read_text())["contract"])
    elif embedded is None:
        import warnings
        warnings.warn("V1历史checkpoint无契约清单，兼容性尚未验证")

"""扩展实验记录与契约校验；优化循环和权重保存仍由上游执行。"""
import json
import os
from pathlib import Path
import torch
from mjlab.rl import MjlabOnPolicyRunner
from rebotarm_rl.contracts.artifacts import file_hash, write_run_record
from .validation import validate_config


class RecordedRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.contract = validate_config(env.unwrapped.cfg)
        obs = env.get_observations()
        self.contract.validate_shapes(obs["actor"].shape[-1], env.num_actions)
        super().__init__(env, train_cfg, log_dir, device)
        # 固定RSL-RL版本没有关闭代码快照的公开配置，仅替换此日志钩子。
        self.logger._store_code_state = lambda: []
        if log_dir and getattr(self, "global_rank", 0) == 0:
            path = Path(log_dir)
            write_run_record(path, repository=Path(__file__).resolve().parents[4],
                runtime={"backend": "mjlab", "device": str(device),
                         "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                         "seed": env.unwrapped.cfg.seed})
            if os.environ.get("REBOTARM_RL_SAVE_COMPILED_MODEL") == "1":
                import mujoco
                mujoco.mj_saveModel(env.unwrapped.sim.mj_model,
                                   str(path / "compiled_model.mjb"), None)
        self._record_directory = Path(log_dir) if log_dir else None
        self._validation_protocol = None
        if log_dir and os.environ.get('REBOTARM_RL_VALIDATION'):
            from .evaluation.selection import validate_protocol, write_json
            if int(os.environ.get('WORLD_SIZE', '1')) != 1:
                raise ValueError('固定验证目前仅支持单进程训练')
            self._validation_protocol = validate_protocol(json.loads(os.environ['REBOTARM_RL_VALIDATION']))
            if self._validation_protocol['interval'] % train_cfg['save_interval'] != 0:
                raise ValueError('验证间隔必须为权重保存间隔的整数倍')
            write_json(Path(log_dir) / 'validation_protocol.json', self._validation_protocol)

    def save(self, path: str, infos=None) -> None:
        """随权重保存契约；保留上游环境计数器与上传逻辑。"""
        super().save(path, {**(infos or {}), "policy_contract": self.contract.to_dict()})
        self._last_saved_checkpoint = Path(path)
        if (self._validation_protocol is not None
                and int(Path(path).stem.removeprefix('model_')) > 0
                and int(Path(path).stem.removeprefix('model_')) % self._validation_protocol['interval'] == 0):
            from .evaluation.selection import validate_saved_checkpoint
            validate_saved_checkpoint(Path(path), self.contract.task_id, self._validation_protocol)

    def learn(self, *args, **kwargs):
        result = super().learn(*args, **kwargs)
        if self._validation_protocol is not None:
            from .evaluation.selection import validate_saved_checkpoint, test_selected_checkpoint
            checkpoint = self._last_saved_checkpoint
            validate_saved_checkpoint(checkpoint, self.contract.task_id, self._validation_protocol)
            test_selected_checkpoint(checkpoint.parent, self.contract.task_id, self._validation_protocol)
        return result

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

    仅对用户信任的checkpoint使用PyTorch加载，必须包含匹配的内嵌契约。
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    embedded = (payload.get("infos") or {}).get("policy_contract")
    if embedded is not None:
        contract.require_compatible(embedded)
    else:
        raise ValueError("拒绝无内嵌契约的checkpoint；请使用当前任务重新训练")
    record_path = Path(path).parent / "run_manifest.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        # 旧运行记录可能带契约；新记录只从权重读取契约。
        if "contract" in record:
            contract.require_compatible(record["contract"])
        elif record.get("schema_version") != 2:
            raise ValueError("未知运行记录格式")

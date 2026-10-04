# rebotarm_rl

reBotArm强化学习项目，使用mjlab + MuJoCo Warp + RSL-RL训练机械臂Reach任务，并提供CPU/GPU配对评估。当前策略输出六关节力矩。

## 安装

适用于Ubuntu、Python 3.12和NVIDIA GPU。依赖文件固定了CUDA 13组合，安装前需确认驱动兼容。

```bash
git clone https://github.com/huangbinai/rebotarm_rl.git
cd rebotarm_rl
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/mjlab-cu130.txt
python -m pip install -e .
```

模型XML和STL随仓库保存在`src/rebotarm_rl/assets/rebotarm/`，克隆后直接可用，并随Python包安装。加载时按`model_manifest.json`校验SHA-256，无需另行下载或配置缓存。

## 安装验证

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
python -m mjlab.scripts.list_envs
```

任务列表应包含`RebotArm-Reach-Mjlab`。环境问题见[故障排查](docs/development.md#故障排查)。

## 训练

正式训练前先提交代码并保持工作区干净；每次记录提交号，不复制Git差异或源码。

```bash
MUJOCO_GL=egl python -m mjlab.scripts.train RebotArm-Reach-Mjlab --env.scene.num-envs 128 --agent.max-iterations 1000 --log-root runs/train
```

首次运行可先用`bash configs/experiments/reach_smoke.sh`完成短训练。短训练用于检查运行链路，不代表策略已经收敛。

## 回放与评估

```bash
python -m mjlab.scripts.play RebotArm-Reach-Mjlab --checkpoint-file /path/to/model.pt --env.scene.num-envs 1
MUJOCO_GL=egl python -m rebotarm_rl.evaluation.paired_eval --task RebotArm-Reach-Mjlab --checkpoint /path/to/model.pt --episodes 100 --steps 250 --seed 20000 --output /path/to/run/eval/model_seed20000.json
```

仅加载可信来源且策略契约匹配的权重。观测、动作单位与权重兼容性见[策略契约](docs/policy_contract.md)。

## 文档

- [训练记录与产物](docs/experiments.md)：输出内容、环境快照、评估选项与清理规则。
- [开发规范](docs/development.md)：验证、模型资源、实验记录与故障排查。
- [结构规范](docs/architecture.md)：目录职责与依赖边界。
- [云端训练](docs/cloud_training.md)：训练环境与产物管理。
- [代理规则](AGENTS.md)：代码代理的执行约束。

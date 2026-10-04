# rebotarm_rl

reBotArm独立强化学习项目。目前实现mjlab + MuJoCo Warp + RSL-RL的Reach训练，以及CPU/GPU配对评估。Isaac Lab尚未实现。训练不依赖ROS工作区、ROS运行时、电机SDK或实机连接。

## 快速开始

适用于Ubuntu、Python 3.12和NVIDIA GPU。安装入口为`requirements/mjlab-cu130.txt`；它固定已验证的CUDA 13依赖组合，换机器时需要检查驱动兼容性。未来Isaac Lab使用独立环境。

```bash
git clone https://github.com/huangbinai/rebotarm_rl.git
cd rebotarm_rl
python3.12 -m venv .venv
source .venv/bin/activate
# 清除继承自ROS的路径，避免污染训练环境。
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
python -m pip install -r requirements/mjlab-cu130.txt
python -m pip install -e '.[test]'
python -m rebotarm_rl.resources
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
python -m mjlab.scripts.list_envs
MUJOCO_GL=egl python -m mjlab.scripts.train RebotArm-Reach-Mjlab-V2 --env.scene.num-envs 128 --agent.max-iterations 1000 --log-root runs/mjlab-v2
```

新实验使用V2任务，其姿态观测为标准wxyz相对旋转向量。历史任务`RebotArm-Reach-Mjlab`保留V1输入语义。两者维度相同，但权重不能混用；V2需要重新训练。

模型获取器仅下载清单列出的文件，来源固定到robotarm_ros2不可变提交，并逐文件校验SHA-256。缓存路径为`$XDG_CACHE_HOME/rebotarm_rl/models/<commit>`，默认在`~/.cache`。训练不会隐式下载，基线资源缺失或被修改会报错。

ROS仓库仍是模型来源；采用新模型时显式更新清单，不手工修改缓存。也可从完全匹配的离线模型包导入：

```bash
python -m rebotarm_rl.resources --source /path/to/model-bundle
```

自定义实验可设置`REBOTARM_MJLAB_SCENE=/path/to/reach_scene.xml`，需保留XML引用和网格文件。自定义模型不执行基线哈希匹配，实验清单会记录实际场景及编译模型哈希；复现实验时应同时保存完整资源。

## 回放与评估

```bash
python -m mjlab.scripts.play RebotArm-Reach-Mjlab-V2 --checkpoint-file /path/to/model.pt --env.scene.num-envs 1
MUJOCO_GL=egl python -m rebotarm_rl.evaluation.paired_eval --task RebotArm-Reach-Mjlab-V2 --checkpoint /path/to/model.pt --episodes 100 --steps 250 --seed 20000 --output runs/paired_eval_v2.json
```

V1历史权重请显式选用`RebotArm-Reach-Mjlab`。评估命令省略`--task`时仍默认V1，以保持历史命令兼容。仅加载可信来源的权重文件。

短训练只验证执行链路，不代表收敛或实机可用。当前动作是六关节力矩，不能当作ROS位置轨迹。部署前阅读[策略契约](docs/policy_contract.md)。

## 结构与协作

- `src/rebotarm_rl/backends/mjlab/tasks/`：任务观测、动作、奖励与终止。
- `src/rebotarm_rl/training/rsl_rl/`：网络和PPO配置，算法实现由上游维护。
- `src/rebotarm_rl/backends/mjlab/evaluation/`：固定目标的CPU/GPU比较。
- `src/rebotarm_rl/assets/`：固定版本资源获取及完整性校验。
- `requirements/`：安装依赖与迁移环境快照。
- `tests/`：契约、资源及后端集成测试。
- `docs/`：结构、开发、云训练与迁移记录。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -I -m pytest tests -q
python -m compileall src/rebotarm_rl -q
```

保留V1短训练脚本`configs/experiments/reach_smoke.sh`；V2使用`configs/experiments/reach_v2_smoke.sh`和`configs/evaluation/reach_v2_smoke.sh`。基础包无强制仿真依赖；训练需安装固定GPU依赖或mjlab可选依赖。

使用分支和PR协作。CPU CI检查契约和资源，GPU训练与评估需要GPU执行器。凭据、权重、缓存和视频不提交Git；训练产物存入对象存储并保留代码版本、模型、依赖和种子。

本仓库通过`git subtree split`从robotarm_ros2提取，保留原`rebotarm_rl/`子树历史。可作为独立Codex项目管理。详见[代理规则](AGENTS.md)、[结构规范](docs/architecture.md)和[开发规范](docs/development.md)。

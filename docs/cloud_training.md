# 云端训练

1. 准备与固定CUDA依赖兼容的Linux GPU机器。
2. 克隆仓库并固定提交，创建独立Python 3.12环境。
3. 安装requirements/mjlab-cu130.txt和本项目，获取固定版本模型。
4. 检查CUDA和任务注册，先执行无窗口短训练。
5. 通过作业调度器或终端会话管理器运行长任务，将权重和日志写入持久存储。
6. 每次实验记录提交、模型清单、依赖、GPU/驱动、种子、命令和指标，并保存归一化及环境配置。
7. 发布策略前使用固定目标与种子评估。

凭据通过云平台身份和密钥系统提供，不写入仓库。云机器无需克隆或构建ROS项目；资源获取器直接下载固定版本模型。

```bash
MUJOCO_GL=egl python -m mjlab.scripts.train RebotArm-Reach-Mjlab-V2 --env.scene.num-envs 8 --agent.max-iterations 1 --log-root runs/smoke-v2
```

迁移环境快照用于诊断，不是跨平台锁文件；安装入口为固定CUDA依赖文件。正式部署应构建测试过的容器并固定镜像摘要。

Isaac Lab实现时在本仓库增加独立后端及任务，固定兼容的Isaac Lab、Isaac Sim与容器版本，运行环境与mjlab分离。复用机器人参数来源及策略契约，不假设物理或权重可互换。目前未实现Isaac任务和云部署。

## 实验产物

原生mjlab训练命令自动使用RecordedRunner。每次运行生成run_manifest.json、model_manifest.json、dependencies.txt、compiled_model.mjb及原生params/env.yaml、params/agent.yaml。上传时与权重一同保留；自定义模型需另存完整源资源。

V2权重内嵌契约，恢复训练禁止跨版本。工作区修改状态、恢复来源和V1历史权重规则见[开发规范](development.md)。短训练和本地测试不代表云部署或收敛已验证。

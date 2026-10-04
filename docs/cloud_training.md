# 云端训练

1. 准备与固定CUDA依赖兼容的Linux GPU机器。
2. 克隆仓库并固定提交，正式训练保持工作区干净，创建独立Python 3.12环境。
3. 安装requirements/mjlab-cu130.txt和本项目；模型资源已随仓库提供。
4. 检查CUDA和任务注册，先执行无窗口短训练。
5. 通过作业调度器或终端会话管理器运行长任务，将权重和日志写入持久存储。
6. 每次实验保留代码版本、命令、种子、最终配置、指标与权重；完整依赖按环境保存。
7. 发布策略前使用固定目标与种子评估。

凭据通过云平台身份和密钥系统提供，不写入仓库。模型随仓库和安装包分发，无需额外下载。

```bash
REBOTARM_RL_RUN_KIND=smoke MUJOCO_GL=egl python -m mjlab.scripts.train RebotArm-Reach-Mjlab --env.scene.num-envs 8 --agent.max-iterations 1 --log-root runs/smoke
```

安装入口为固定CUDA依赖文件。正式部署应构建测试过的容器并固定镜像摘要。

当前训练后端为mjlab，后端扩展要求见[结构规范](architecture.md)。本文是运行指南，不代表已验证远程云部署。

## 实验产物

按[训练记录与产物](experiments.md)保存运行目录与环境快照。上传后校验完整性再清理本地副本。自定义模型另存完整资源，云环境补充驱动信息或容器镜像摘要。

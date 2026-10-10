# 代码框架与结构规范

本项目负责仿真训练、评估与策略制品。
当前只有 mjlab 后端；Isaac Lab、自定义算法、导出和云调度器按需求新增，
不创建占位类或空后端。

## 结构

- src/rebotarm_rl/contracts：标准库实现的机器人、策略与实验记录契约。
- assets：随包模型XML/STL、来源清单及哈希校验。
- backends/mjlab/robots：模型与执行器配置。
- backends/mjlab/tasks/reach：配置、命令、观测、奖励、终止。
- backends/mjlab/registration：唯一插件注册点。
- backends/mjlab/runner：配置/输入输出校验和实验记录；优化循环仍由上游执行。
- backends/mjlab/evaluation：依赖引擎的 CPU/GPU 配对评估。
- training/rsl_rl：RSL-RL 网络和算法配置，不复制上游算法。
- scripts/train.py：标准库实现的仓库启动入口，读取实验覆盖后交给mjlab原生CLI。
- scripts/play.py：标准库隔离启动入口；backends/mjlab/play.py负责固定惩罚任务的回放配置和原生窗口显示，复用原生runner和viewer，不修改训练任务或物理模型。
- configs/experiments：命名实验TOML，不复制任务或算法实现。
- configs/evaluation：可复用评估命令。
- tests/unit、integration：资源与记录逻辑、策略契约及后端集成测试。

未来公共指标放 evaluation；只有独立于引擎的计算才能提取。
policies 和 algorithms 等到出现自定义网络或算法实现后再创建。

## 依赖规则

训练代码不依赖ROS运行时、其他工作区的Python包或硬件驱动。
contracts/assets 不导入 torch、mjlab、mujoco 或 ROS。
Isaac Lab尚未实现，未来与mjlab使用独立运行环境，不互相导入或调用环境；可复用公共契约，不假设物理或权重可互换。
顶层 import 不加载模型、初始化 GPU 或注册任务；插件注册是显式边界。
环境和模型生命周期归后端；算法函数不创建仿真。
模型资源随仓库和Python包分发，不使用相邻工作区或外部缓存路径。

## 兼容边界
保持 RebotArm-Reach-Mjlab 任务名与 console scripts。
仅为已公开的 python -m rebotarm_rl.resources 和
python -m rebotarm_rl.evaluation.paired_eval 保留薄命令入口。
旧 tasks/agents Python 内部导入路径不保留兼容层。
任务语义不随目录迁移改变。改变动作或观测必须升级契约并明确checkpoint兼容性；当前`reach-effort-v1`已停止作为正式任务，正式任务为`reach-position-v1.1`。

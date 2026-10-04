# 代码框架与结构规范

本项目负责仿真训练、评估与策略制品，不负责 ROS、硬件通信或实机控制。
当前只有 mjlab 后端；Isaac Lab、自定义算法、导出和云调度器按需求新增，
不创建占位类或空后端。

## 结构
- src/rebotarm_rl/contracts：标准库实现的机器人、策略与实验记录契约。
- assets：固定版本模型清单、获取及哈希校验。
- backends/mjlab/robots：模型与执行器配置。
- backends/mjlab/tasks/reach：配置、命令、观测、奖励、终止。
- backends/mjlab/registration：唯一插件注册点。
- backends/mjlab/runner：配置/输入输出校验和实验记录；优化循环仍由上游执行。
- backends/mjlab/evaluation：依赖引擎的 CPU/GPU 配对评估。
- training/rsl_rl：RSL-RL 网络和算法配置，不复制上游算法。
- configs：可复用实验与评估命令配置。
- tests/unit、contracts、integration：纯逻辑、架构契约及后端集成测试。

未来公共指标放 evaluation；只有独立于引擎的计算才能提取。
policies 和 algorithms 等到出现自定义网络或算法实现后再创建。

## 依赖规则
contracts/assets 不导入 torch、mjlab、mujoco 或 ROS。
mjlab 不导入 Isaac Lab。未来两个后端复用契约，不互相调用环境。
顶层 import 不加载模型、初始化 GPU 或注册任务；插件注册是显式边界。
环境和模型生命周期归后端；算法函数不创建仿真。
资源不使用相邻工作区路径。模型缓存不在源码仓库内维护。

## 兼容边界
保持 RebotArm-Reach-Mjlab 任务名与 console scripts。
仅为已公开的 python -m rebotarm_rl.resources 和
python -m rebotarm_rl.evaluation.paired_eval 保留薄命令入口。
旧 tasks/agents Python 内部导入路径不保留兼容层。
任务语义不随目录迁移改变。改变动作或观测必须升级契约并明确checkpoint兼容性。

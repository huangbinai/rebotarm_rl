# 001：公共契约与独立后端
状态：已实施首轮，2026-10-04。

采用src布局，mjlab任务与配对评估归后端，训练参数归training/rsl_rl。
使用标准库定义公共契约。依赖通过mjlab extra按需安装，避免未来Isaac环境
被强制安装另一套引擎。不构建通用环境父类，不复制PPO，不实现空Isaac目录。

通过mjlab runner_cls扩展点记录实验，原生train/play命令保持可用。
保留两个已公开python -m薄入口；内部模块路径迁移，不增加全面兼容层。

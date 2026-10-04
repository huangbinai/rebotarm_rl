# 001：公共契约与独立后端
状态：已实施首轮，2026-10-04。

采用src布局，mjlab任务与配对评估归后端，训练参数归training/rsl_rl。
使用标准库定义公共契约。依赖通过mjlab extra按需安装，避免未来Isaac环境
被强制安装另一套引擎。不构建通用环境父类，不复制PPO，不实现空Isaac目录。

通过mjlab runner_cls扩展点记录实验，原生train/play命令保持可用。
保留两个已公开python -m薄入口；内部模块路径迁移，不增加全面兼容层。
既有姿态误差编码保留，不能在结构改造中静默修正。


## 验证结果
- 独立mjlab环境：10项测试通过。
- wheel在仓库外、无训练框架环境安装：9通过、1后端集成测试跳过。
- GPU短训练：8环境、1迭代完成，checkpoint及实验清单/依赖/模型快照生成。
- 相同短训练的actor、critic和optimizer state_dict与重构前逐张量完全一致。
- 新旧checkpoint各运行2回合、20步CPU/GPU评估；旧checkpoint结果与迁移前
  报告逐回合完全一致。短训练成功率仍为0，不能视为策略收敛。
- 4个观测/奖励/终止函数AST一致；compileall、shell语法、diff检查通过。
- 首次检查发现旧editable元数据残留，已移出源码并重新安装；
  契约维度按实际运行确认为18/6。MjSpec模型不支持mj_saveLastXML，
  编译模型证据改为MJB。
- 保留历史姿态观测编码，明确其不等于标准wxyz相对旋转；
  该问题应作为独立任务语义升级处理。
- 仅本地验证；GitHub CI、Isaac Lab、云端及实机未运行。

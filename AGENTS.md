# 代码代理工作规则

先阅读README.md、docs/policy_contract.md、docs/cloud_training.md、docs/MIGRATION.md、docs/architecture.md和docs/development.md。

这是独立训练仓库，不是ROS包。禁止导入rclpy、rebotarm_simulation、rebotarm_preview或rebotarmcontroller。
禁止连接真实硬件。力矩策略不等于位置轨迹。
保持已有mjlab任务名和动作、观测语义；显式修改任务时必须升级版本并说明权重不兼容性。
模型通过固定版本model_manifest.json获取，不依赖相邻工作区。
凭据、权重、视频和虚拟环境不进入Git。
Isaac Lab尚未实现，未来必须与mjlab使用独立运行环境。

必要检查：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -I -m pytest tests -q
python -m compileall src/rebotarm_rl -q
git diff --check
```

改变任务动力学必须执行GPU短训练和配对评估。报告中区分链路验证、策略收敛、泛化能力和实机验收。

# 代码代理工作规则

先阅读README.md、docs/policy_contract.md、docs/cloud_training.md、docs/architecture.md和docs/development.md。

保持训练环境与其他工作区独立，遵守docs/architecture.md中的依赖边界。
禁止连接真实硬件。策略接口以docs/policy_contract.md为准。
保持已有mjlab任务名和动作、观测语义；显式修改任务时必须升级版本并说明权重不兼容性。
模型随仓库保存在src/rebotarm_rl/assets/rebotarm/，通过model_manifest.json校验，不依赖相邻工作区或外部缓存。
凭据、权重、视频和虚拟环境不进入Git。
训练记录遵循docs/experiments.md，优先使用mjlab原生保存能力。
短验证产物仅临时使用：训练、权重加载及所需评估全部检查通过后，由Codex清理对应smoke运行目录，只在对话中汇报结果；失败产物保留至问题解决。清理前确认无运行中任务或仍需使用的引用，用户明确要求保留时除外。
正式训练前先提交代码并保持工作区干净，训练期间不修改源码；每次记录Git提交号，不保存runs内的git代码快照。短验证允许未提交改动，但不能据此宣称可复现。
Isaac Lab尚未实现，未来必须与mjlab使用独立运行环境。

## 分析与说明的保存

日常解释、代码修复和短训练结果只在对话中汇报，不新增报告文件；用户明确要求保存时除外。
由Codex执行的正式训练完成后，读取训练曲线与独立评估结果，在对应运行目录写一份简短的analysis.md，记录结果、问题、结论和下一步，并引用所用权重和评估文件。
缺少独立评估或其他证据时明确标注，不推断收敛、泛化或实机可用性。正式训练失败或中断时记录实际状态，不写成已完成。
同一运行的分析更新已有analysis.md，不逐轮新增报告；接口或运行方式变化时更新已有文档，避免重复新增说明。
这是Codex执行任务时的规则，训练脚本本身不会自动调用AI或生成分析报告。

必要检查：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -I -m pytest tests -q
python -m compileall src/rebotarm_rl -q
git diff --check
```

改变任务动力学必须执行GPU短训练和配对评估。报告中区分链路验证、策略收敛、泛化能力和实机验收。

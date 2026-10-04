# 开发和协作

## 环境与验证
采用src布局，先安装再测试；不要用PYTHONPATH=src掩盖打包缺陷。
基础包不强制安装训练框架，GPU依赖按requirements安装。后端环境边界见[结构规范](architecture.md)。

    python -m pip install -e '.[test]' --no-deps
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -I -m pytest tests -q
    python -m compileall src/rebotarm_rl -q
    git diff --check

纯CPU CI安装基础包后运行unit测试；未安装后端依赖时integration明确跳过。
任务或训练适配变更必须运行GPU短训练与固定种子配对评估，并测试旧checkpoint。
wheel必须包含assets/model_manifest.json及assets/rebotarm/下全部XML和STL；在仓库之外、空缓存环境验证入口。

## 模型资源

模型随仓库保存在`src/rebotarm_rl/assets/rebotarm/`，XML和STL一同纳入Git并打包，加载时逐文件校验SHA-256。`model_manifest.json`保留最初来源与文件校验值，当前运行不从来源仓库下载。
修改基线模型时同步更新清单中的文件哈希，并执行相应动力学验证，不手工修改安装环境中的资源。

可在安装后检查资源完整性并打印场景路径：

```bash
python -m rebotarm_rl.resources
```

旧命令`rebotarm-rl-fetch-model`保留为同一校验入口，不再下载；`--source`离线导入选项已移除。

自定义实验可设置`REBOTARM_MJLAB_SCENE=/path/to/reach_scene.xml`，需保留XML引用和网格文件。自定义模型不执行基线哈希匹配；复现实验时应另行保存场景设置和完整资源。

## 故障排查

### 环境路径污染

如果终端曾加载ROS或其他工作区，激活虚拟环境后仍可能继承外部Python路径。遇到导入冲突或pytest加载外部插件时，先在干净终端重试；必要时清除继承的路径：

```bash
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
```

这是受污染环境的排查步骤，新环境无需执行。

### MuJoCo诊断日志

`MUJOCO_LOG.TXT`是MuJoCo自动生成的诊断日志，已被Git忽略。确认内容后可以删除；再次出现警告时可能重新生成。

当前模型挂接时可能提示父场景与子模型的`integrator`冲突。挂接阶段保留父场景值，随后mjlab通过`MujocoCfg`应用仿真配置；最终积分器应以运行模型或配对评估报告的`integrator`字段为准。删除日志不会消除警告原因，不应通过屏蔽所有警告来处理。

## 代码规则
公共接口写类型、单位、坐标系和副作用。纯函数优先，类用于持有真实状态或生命周期。
不按行数拆分；不建立万能utils，不捕获所有异常后继续训练。
配置归后端原生配置系统；命名实验记录覆盖，不复制任务实现。
位置误差方向是current-target，不得仅按字段名字推断。
V1采用标准相对旋转向量，详见policy_contract.md。

## 实验记录

目录、环境快照、评估选项与清理规则统一见[训练记录与产物](experiments.md)。配置、曲线和权重使用mjlab原生保存能力。

## 团队协作
main保持可安装可验证。功能分支+PR；结构改动和行为改动分开提交。
PR说明问题、接口变化、checkpoint兼容性、测试和未验证范围。
云端镜像/依赖/模型版本固定，凭据通过密钥系统注入。
CI配置不等于CI已经运行；短训练不等于收敛或实机验收。

# 开发和协作

## 环境与验证
采用src布局，先安装再测试；不要用PYTHONPATH=src掩盖打包缺陷。
mjlab/Isaac Lab环境独立。基础包不强制安装训练框架，GPU依赖按requirements安装。

    python -m pip install -e '.[test]' --no-deps
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -I -m pytest tests -q
    python -m compileall src/rebotarm_rl -q
    git diff --check

纯CPU CI运行可安装基础包及unit/contracts；未装mjlab时integration明确跳过。
任务或训练适配变更必须运行GPU短训练与固定种子配对评估，并测试旧checkpoint。
wheel必须包含assets/model_manifest.json；在仓库之外验证入口。

## 代码规则
公共接口写类型、单位、坐标系和副作用。纯函数优先，类用于持有真实状态或生命周期。
不按行数拆分；不建立万能utils，不捕获所有异常后继续训练。
配置归后端原生配置系统；命名实验记录覆盖，不复制任务实现。
位置误差方向是current-target，不得仅按字段名字推断。
保持历史姿态编码直到单独任务变更，详见policy_contract.md。

## 实验记录
RecordedRunner在训练前写入run_manifest.json、model_manifest.json、
dependencies.txt和compiled_model.mjb。
原生params/env.yaml、params/agent.yaml保留最终任务与PPO配置，避免另造序列化器。
记录Git提交、dirty状态、seed、设备、参数、模型及编译模型哈希。
恢复checkpoint记录来源及哈希；有契约时拒绝不兼容版本，
历史checkpoint缺少契约时明确警告，不能宣称兼容已被证明。
产物保留mjlab原生布局，避免破坏恢复与play入口。

本地可使用dirty工作区；正式比较和发布使用干净固定提交。
自定义模型不是固定基线：manifest中的baseline仅表示参考来源，
scene与compiled_model哈希记录实际输入。MJB用于同版本MuJoCo复查；跨版本复现仍需原始资源与编译设置。
checkpoint、日志、视频在runs或对象存储，不能提交普通Git。

## 团队协作
main保持可安装可验证。功能分支+PR；结构改动和行为改动分开提交。
PR说明问题、接口变化、checkpoint兼容性、测试和未验证范围。
云端镜像/依赖/模型版本固定，凭据通过密钥系统注入。
CI配置不等于CI已经运行；短训练不等于收敛或实机验收。

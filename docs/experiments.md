# 训练记录与产物

## 统一训练入口

在独立虚拟环境中运行仓库的`scripts/train.py`，从任意工作目录传入脚本路径均可；入口自动切换到仓库根目录，使用当前Python解释器执行mjlab原生CLI。默认实验为`reach_gravity_fixed`。

```bash
python scripts/train.py --experiment reach_gravity_fixed --seed 7 --dry-run
python scripts/train.py --experiment reach_gravity_fixed --seed 7
python scripts/train.py --experiment reach_gravity_fixed_smoke
```

`configs/experiments/*.toml`只记录命名实验覆盖：`task`、`run_kind`、`seed`和`[args]`。参数表使用带引号的原生CLI键，例如`"agent.max-iterations" = 1000`。目前支持字符串和数值；网络结构等复杂配置仍放在原生配置中。未知原生参数由mjlab拒绝，不自行实现第二套训练配置系统。

新增实验时新增TOML，不复制训练脚本。种子优先级为显式`--seed` > TOML中的`seed`；省略命令行参数时，主线使用TOML记录的seed42，历史配方使用seed7。运行名为`实验名_seed种子`，输出根目录由`run_kind`选择`runs/train`或`runs/smoke`。这些字段由入口管理，不在`[args]`中重复定义。基线使用128环境、24步采样、1000轮更新；短验证使用同样并行规模、2轮更新。旧任务的四个短训练TOML已删除；短验证使用当前固定惩罚任务。

入口固定使用仓库模型，移除继承的ROS/PYTHONPATH和自定义模型路径，设置EGL及实验类型；不修改父终端环境。需要自定义模型或原生复杂参数时直接使用mjlab CLI，并遵守模型与契约规则。

入口在启动GPU前检查正式训练的Git提交与干净状态，后端保留二次检查；`--dry-run`不执行该检查、不启动训练、不写产物。Ctrl+C直接交给原生训练处理。训练期间不要修改源码或实验配置。

使用`--environment 名称`引用`runs/environments/名称.txt`，不存在或为空时拒绝启动。入口不继承`REBOTARM_RL_ENVIRONMENT`，未指定时记录当前Python环境路径，不自动生成快照。快照是否仍匹配当前依赖需要使用者确认；更新依赖后应生成新快照。

```bash
python scripts/train.py --experiment reach_gravity_fixed --environment 2026-10-04-preflight
```

上例仅适用于已存在且仍匹配的快照。运行记录保存展开后的原生命令，最终配置仍由mjlab写入`params/`；实验TOML由Git提交追溯，不另存重复配置。入口默认不自动评估；显式使用--validation启用下文固定验证，分析说明仍由Codex读取结果后维护。

实验目录仅保留当前的 `reach_gravity_fixed` 和 `reach_gravity_fixed_smoke`；五个历史正式配方位于 `configs/experiments/history/`，通过 `--experiment history/名称` 显式选择。运行名仍使用配方文件名，不含 `history/`。历史任务注册和权重契约保持不变。

## 默认输出

官方对齐及其派生 Reach 实验统一写入 `runs/train/rebotarm_reach/`，各次运行仍以时间戳和实验名区分。历史目录 `rebotarm_reach_official_aligned` 已迁移，并更新记录中的权重路径；历史 `params/agent.yaml` 保留当时的实验目录名称，不改写原始训练配置。

已完成的任务1运行按用途整理：`mainline/` 保留固定惩罚8轮PPO的seed7、17、31三个运行，`history/` 保留其余七次基线及消融运行。迁移同步更新评估JSON、恢复来源路径和分析中的相对链接，保留权重内容与哈希。新训练仍由原生训练器写入 `rebotarm_reach/<时间戳>_<运行名>/`，完成评估后再决定归类；目录分类不自动代表训练质量。

使用mjlab原生目录，短训练指定`--log-root runs/smoke`，正式训练指定`--log-root runs/train`。上游在其下创建`<实验名>/<时间戳>[_run_name]/`。恢复训练创建新目录，记录来源权重路径及哈希。

| 文件 | 内容与负责方 |
|---|---|
| `params/env.yaml`、`params/agent.yaml` | mjlab保存最终生效配置 |
| `events.out.tfevents.*` | 原生TensorBoard曲线 |
| `model_*.pt` | 原生训练状态，项目内嵌策略契约 |
| `run_manifest.json` | 提交、dirty状态、Python启动参数、训练种子、设备、环境标识及恢复来源 |
| `eval/*.json` | 评估时生成，包含条件、指标、权重身份及可选选择依据 |

命令记录是Python进程启动参数，不包含shell重定向或完整环境变量。自定义场景等设置需保留资源并在实验说明中注明。
正式训练前提交代码并保持工作区干净，运行记录保留提交号，不再生成运行目录内的`git/`。项目根目录`.git/`必须保留。
默认按正式训练检查；无有效提交、有未提交改动或未跟踪文件时拒绝开始训练（上游可能已创建配置目录）。
短验证脚本自动设置`REBOTARM_RL_RUN_KIND=smoke`，允许未提交改动；直接运行短验证需显式设置该变量。类型由变量决定，不按目录名或迭代次数猜测。短验证不备份代码改动，不作为可复现实验依据。
训练期间不要修改源码，正式产物归档时保留对应提交。项目通过固定版本RSL-RL的日志钩子关闭代码快照，升级依赖时需验证该钩子。

## 环境和模型

完整依赖仅在环境建立或更新时导出，按日期存放，更新时创建新文件，不覆盖已有快照。例如：

```bash
mkdir -p runs/environments
python -m pip freeze --all > runs/environments/2026-10-04.txt
export REBOTARM_RL_ENVIRONMENT=2026-10-04
```

名称需与实际快照对应；同一天更新时增加后缀。训练读取该环境标识，未设置时记录Python环境路径，不自动导出依赖。归档时同时携带对应环境文件；需要严格复现时补充驱动信息或容器镜像摘要。

固定模型随项目的`assets/rebotarm/`分发，由`assets/model_manifest.json`校验，运行目录不重复复制来源清单。默认不保存编译模型；排查或正式归档时显式设置：

```bash
REBOTARM_RL_SAVE_COMPILED_MODEL=1 python scripts/train.py --experiment reach_gravity_fixed_smoke
```

此时在运行目录生成`compiled_model.mjb`。MJB用于同版本MuJoCo复查，不代表完整训练状态或跨版本精确复现。自定义场景需另存完整XML、网格等源资源。本项目不建立模型去重或共享引用系统。

## 评估

将结果写回所属运行目录，名称包含权重编号和评估种子：

```bash
python -m rebotarm_rl.evaluation.paired_eval \
  --checkpoint /path/to/run/model_999.pt \
  --episodes 100 --steps 250 --seed 20000 \
  --output /path/to/run/eval/model_999_seed20000.json
```

报告记录权重路径及哈希、任务契约、种子、回合与步数、目标采样、每回合初始关节状态、动作处理、终止规则、成功阈值、后端、成功次数和误差。默认不输出逐步轨迹；正式实验或失败诊断使用`--save-trajectories`保留完整误差轨迹。选定权重时用`--selection-reason "选择标准及比较依据"`说明原因，不复制另一份selected权重。未指定表示仅评估，程序不会自动选择最佳权重。

配置以params为准，曲线以TensorBoard为准，评估以JSON为准；不强制生成summary.md或重复CSV。训练指标是训练采样统计，不能代替独立评估。CPU/GPU一致性不证明策略有效性；泛化需独立测试条件，实机验收不在此流程中。

## 分析说明

日常解释、代码修复和短训练结果只在对话中汇报，不新增报告文件，除非用户明确要求保存。
由Codex执行的正式训练完成后，读取训练曲线与独立评估结果，在该运行目录维护一份简短的`analysis.md`，包含结果、问题、结论和下一步，并引用所用权重及评估文件，不复制完整配置、曲线或轨迹。
缺少独立评估或其他证据时明确标注，不推断收敛、泛化或实机可用性；失败或中断时如实记录状态。同一运行更新原文件，不逐轮新增报告。
接口或运行方式变化时更新已有项目文档。直接在终端运行训练不会自动调用AI或生成`analysis.md`。

## 保存与清理

使用原生`--agent.save-interval`控制保存频率，当前任务默认100次迭代；最终权重由框架保存。默认不录制视频。

- 短训练：产物仅临时使用。训练、权重加载及所需评估全部检查通过后，清理对应运行目录，结果只在对话中汇报。失败产物保留至问题解决；用户明确要求保留时除外。不能仅凭训练进程正常退出就提前删除后续评估需要的权重。
- 正式实验：保留代码、命令、种子、配置、曲线、最终可恢复权重、选定权重和结论引用的权重；相同文件不重复存放。保留正式评估条件和证据。
- 清理前检查`du -h --max-depth=2 runs`，列出待删除项，确认相关训练、恢复和评估已结束且无仍需使用的引用后执行。Codex按本规则清理已完成的短验证，无需每次再请求确认；训练脚本不自动删除产物，用户自行运行后需自行清理。
- 环境快照被实验引用时保留；`runs/`整体不提交Git，正式产物归档到持久存储。

历史runs单独盘点，不因记录格式调整自动搬迁或删除。仓库只维护本规范，不为每次短训练增加验证报告。

## PPO更新轮数对照

`reach_gravity_fixed_epochs4`沿用固定惩罚任务，仅将每批数据的PPO学习轮数从8改为4。动作、观测、动力学与奖励契约不变，因此不新增任务或策略契约；训练配方由实验名、提交与params区分。对照从头训练1000轮，seed7、17、31，128环境×24步。

```bash
python scripts/train.py --experiment history/reach_gravity_fixed_epochs4 --seed 7 --environment 2026-10-04-preflight
```

统一用seed30000比较100、200、250、300、400、600、800、999轮；按末段保持成功数最高选候选，并列按平均位置误差最低，再并列选较早轮次。随后在新seed80000的100个目标上比较三种子的候选及最终权重与8轮更新对照。该比较控制采样步数，不等计算量。


## 定期单后端验证与最终独立测试

```bash
python scripts/train.py --experiment reach_gravity_fixed --seed 7 --environment 2026-10-04-preflight --validation reach_validation
```

`--validation`读取`configs/evaluation/<名称>.toml`。默认`interval=100`，跳过编号0，在编号100、200等已保存权重上启动独立GPU评估子进程；训练正常结束后补评最终权重，选定候选，再对候选执行一次不同目标种子的GPU测试。周期沿用原生checkpoint编号（从0计数），不是精确的第100次优化更新。验证间隔必须是原生save-interval的整数倍，不改变保存频率。短训练不足一个周期时仍会评估最终权重。

默认每次100回合×250控制步，固定目标2–6cm，不自动结束或刷新；验证seed30000，独立测试test_seed100000，必须不同。用户重复调参后应主动更换尚未用于决策的test_seed，不能把反复查看的测试集称为永远未见。单GPU后端避免CPU配对轨迹开销，但逐回合评估仍增加运行时间，不增加训练采样步数。独立进程不重置训练环境、不改变父进程随机状态，暂不支持多GPU分布式训练。

协议在`validation_protocol.json`；验证JSON/日志在`eval/validation/`，`selection.json`记录已评估候选和best。按GPU末段保持成功数最高、平均最终位置误差最低、较早轮次选择，只代表已评估检查点中的最佳。末段25样本端点跨度0.48秒。独立测试JSON/日志在`eval/test/`，引用选定权重哈希，测试结果不回写selection，不自动重新选择候选。两类评估均保留逐步误差；脚本不会生成AI分析报告。

评估失败则报错并保留产物，失败结果不会替换best。训练中断或训练异常不自动执行最终独立测试；若训练完成但独立测试失败，应区分两者状态。协议或权重身份不匹配明确拒绝。同一权重同一哈希已评估则跳过，已评估权重被覆盖则拒绝复用。没有复制selected权重、不删除旧权重、不自动早停。未指定--validation时仍保持原训练行为。

评估使用注册任务的环境与网络定义，不能任意覆盖训练动力学、观测或网络结构后假设评估自动复用这些覆盖。PPO学习轮数等不影响推理结构的覆盖可用。旧版selection协议不会静默迁移为本协议。

CPU/GPU配对继续作为手动检查工具；修改动力学或评估实现后重点执行：

```bash
MUJOCO_GL=egl python -m rebotarm_rl.evaluation.paired_eval --task RebotArm-Reach-GravityComp-FixedPenalties-Mjlab --checkpoint /path/to/model.pt --output /path/to/paired.json --backend paired
```

原生入口默认仍是paired；`--backend gpu`仅运行GPU策略轨迹，仍用CPU读取编译模型的初始TCP以保持目标采样一致，不执行CPU策略轨迹。单后端报告的一致性差值为null，不表示差值为零。

短验证：`python scripts/train.py --experiment reach_gravity_fixed_smoke --validation reach_validation_smoke`（2回合，含最终验证及独立测试，仅检查链路）。

## 任务2：现有候选的随机起点诊断

先评估任务1固定惩罚版的三个既定候选，再决定是否新增训练任务。
诊断协议为`task2-random-start-diagnostic-v1`，不会修改旧任务或checkpoint契约。

```bash
MUJOCO_GL=egl python -m rebotarm_rl.backends.mjlab.evaluation.random_start \
  --checkpoints /path/to/seed7/model_250.pt /path/to/seed17/model_600.pt /path/to/seed31/model_200.pt \
  --amplitudes 0 .025 .05 .1 .2 .4 --episodes 100 \
  --target-seed 130000 --initial-seed 130001 --paired-episodes 3
```

每档六关节独立均匀扰动，速度及上一动作为零；从编译模型读取限位，保留
0.01rad余量，并用CPU `mj_forward`拒绝启用碰撞对中接触距离≤0的初态。
按回合独立随机流进行拒绝采样，不裁剪关节角；不同幅度共用归一化候选偏移，
拒绝后的样本可能不同，报告记录拒绝次数及最终初态。
模型现有相邻连杆、结构重叠及手指碰撞排除保持不变；检查不覆盖被排除的几何对。

目标始终以默认姿态FK为中心、径向均匀2–6cm，朝向固定默认值，每回合固定目标
250步（5秒），取消自动结束及刷新。三个候选和各幅度共用相同目标，随机起点
不会移动目标中心或朝向。任务1训练分布是0–6cm，此处沿用其严格评估分布。
初始已成功的回合不重采样目标，另报排除它们后的末段成功数与分母。

成功要求位置<1cm、姿态<3°；末段26样本跨度0.50秒，区别于旧报告25样本的
0.48秒。记录首次到达、首次连续保持完成时间（只对成功者求均值）、末段TCP
相对均值位置的RMS抖动、末段速度RMS、逐关节角度范围/累计行程及最大偏移。
同时用CPU检查全部50Hz轨迹关节样本的限位及启用碰撞对；未检查物理子步，
不能据此宣称连续路径无碰撞。CPU/GPU配对仅覆盖各档前`paired-episodes`回合。

JSON保存在每个原训练运行的`eval/`下，含源码提交及dirty状态、评估器哈希、
权重哈希、条件、逐回合误差轨迹和指标；已有同名文件拒绝覆盖。
来源信息在GPU启动前采集，安装wheel时Git信息明确为null，并记录包版本、源码文件及模型清单哈希。
启动即写`*.incomplete.json`，逐档保留进度；异常时状态为failed并保留原因，
只有全部完成才原子发布`status=completed`的正式JSON。历史v1报告没有status/role字段时仍可读取，
但明确失败或未完成的报告不能用于冻结选择。
`--output-dir`仅用于临时链路核对，检查通过后按短验证规则清理。

本轮先以相对零扰动末段成功率下降≥10个百分点作为明显退化诊断线；最佳候选
在±0.05rad仍≥95%时继续扩大起点扰动，未达标时才针对随机起点训练。
这些是诊断决策线，不能代替正式任务验收。所有用于判断幅度的样本属于验证集；
冻结候选及幅度后才使用新目标/初态种子的独立测试，不能据测试重新选择候选。
若新增训练，另注册任务和契约版本，保持奖励及PPO配方，使用固定目标回合与
经检查的随机reset，并配置独立验证/测试集；不得改写旧权重元数据绕过契约。

冻结后复核可传入`--selection-file /path/to/frozen_selection.json`。文件在测试前
保存`checkpoint_sha256`、选择依据、验证文件引用、`validation_target_seeds`、
`validation_initial_seeds`及`test_settings`；后者包含`amplitudes_rad`、`episodes`、
`steps`、`hold_samples`、`target_seed`、`initial_seed`。入口逐项验证权重与条件，
拒绝复用验证种子；报告保存该冻结文件的内容和哈希并标记`frozen_candidate_test`。
核验包括协议版本、实际验证报告及权重哈希、验证条件、回合完整性、实际种子与声明一致，
以及候选对测试幅度的验证覆盖。轨迹中同时发生越限与碰撞时分别计数；初态拒绝采样仍保留首个拒绝原因，历史报告不回写。
测试不会更新冻结文件或重新选择权重。未传此参数的扫描始终属于诊断验证。

# Reach策略接口与版本

## 独立官方配方对齐实验

`RebotArm-Reach-OfficialAligned-Mjlab` / `reach-position-official-aligned-v2`
保留旧任务，权重不能互换。动作改为 `q_target = q_default + 0.5 * action`，
无任务级动作裁剪，仍受编译模型执行器限制。保留24维误差观测及0.02秒控制周期。
12秒回合，仅超时终止，每4秒刷新目标；固定默认姿态FK为采样中心，
保留0至0.06米径向均匀采样及固定默认姿态，不复制Franka工作空间。
这避免在reset尚未forward时读取旧TCP缓存，也避免刷新时目标中心跟随策略漂移。
保留本机128环境、reBotArm模型/PD增益、无观测噪声与默认reset，
不是官方Franka/PhysX的完整复现。

奖励五项初始权重保持不变；超过4500控制步后，在回合重置时将动作率惩罚设为
-0.005、关节速度惩罚设为-0.001（阶跃，不是线性渐变）。
PPO采用64×64 ELU、初始标准差1.0、熵系数0.001、学习率0.001、8 epochs，
其余采样24步、4 minibatches、gamma 0.99、lambda 0.95、KL 0.01。
参考官方 `main` 的 reach_env_cfg.py、config/franka/joint_pos_env_cfg.py 及
config/franka/agents/rsl_rl_ppo_cfg.py；核对日期2026-10-06。
https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach

启动：`python scripts/train.py --experiment reach_official_aligned`；
短验证使用 `reach_official_aligned_smoke`。评估必须显式传入新任务名，
使用固定目标250步测试（不刷新目标、不自动终止），区别于12秒训练任务。

## 姿态权重单变量实验

`RebotArm-Reach-OfficialAligned-Orientation-Mjlab` /
`reach-position-official-aligned-v2-orientation-v1`只将姿态误差奖励权重从
`-0.1`改为`-0.2`，其他动作、PPO、目标、episode、curriculum和观测保持一致。
它只与自身权重兼容，不能加载官方对齐版v2权重。正式实验：
`python scripts/train.py --experiment reach_orientation_aligned`。

任务配置以`src/rebotarm_rl/backends/mjlab/tasks/reach/config.py`为准，契约定义在`contracts/policy.py`。

## 共同约定

- 动作：joint1至joint6的六个相对关节位置增量，单位rad，缩放0.25；由仿真位置伺服器转换为受限力矩。
- 时序：物理步长0.002秒，每10步执行一次策略，即动作周期0.02秒。
- 观测：六个相对关节位置、六个相对关节速度、三维位置误差、三维姿态误差、六个上一动作，共24维。Actor与Critic使用相同观测组。
- 位置误差：世界系current减target，单位m。
- 四元数：MuJoCo/mjlab使用wxyz，ROS消息使用xyzw，禁止直接复制数组。
- 成功条件：位置误差小于0.01m且姿态角误差小于0.05236rad。奖励和成功指标始终使用标准四元数角差。
- 目标：初始TCP附近的位置，姿态保持初始姿态；现有orientation_radius字段尚未参与目标采样。

## Reach V1任务

任务名`RebotArm-Reach-Mjlab`，契约`reach-position-v1.1`。策略输出相对当前位置的关节目标增量，不直接输出力矩；仿真使用与模型力矩上限一致的PD位置伺服器。
姿态观测为世界系`current * inverse(target)`的最短旋转向量，单位rad，长度为旋转角。它描述从目标姿态旋转到当前姿态的误差；交换两者时在非π边界处符号反转。

实现先归一化有限非零四元数，再选择最短弧；q与-q等价。恰好π时固定绝对值最大轴分量为正，π附近存在旋转对数映射固有的不连续。接近零时使用极限展开避免除零。GPU观测和CPU配对评估调用同一实现，测试使用独立旋转矩阵参照。

实验目录为`rebotarm_mjlab_reach`。

## 权重与部署边界

保存权重时写入内嵌policy_contract，加载前检查完整内嵌契约；旧run_manifest.json若含契约也需匹配。拒绝无内嵌契约或任一契约字段不匹配的权重；相同任务名和24/6维度不代表兼容。此前生成的`reach-effort-v1`权重与当前契约不兼容，需要重新训练，不能通过修改元数据绕过检查。

部署必须保留动作增量缩放、关节顺序、时序、目标坐标系和模型来源。本仓库不包含ROS发布器、Action客户端或硬件驱动。未来部署适配器需要验证反馈时效、单位、限幅和硬件支持的控制模式。位置策略也不能未经验证直接发送给FollowJointTrajectory。

CPU/GPU比较使用mjlab编译模型，不代表与ROS位置控制仿真或实机等效。

## V1.1 修复

控制步开始时采样当前位置，计算 `q_target = q_current + clip(0.25 * action, -0.25, 0.25)`，在10个物理子步中保持该目标。上一动作观测为训练器可选裁剪后的未缩放动作，CPU/GPU保持一致。

精细位置奖励为 `1 - tanh(distance / 0.10)`，权重0.1；叠加权重-0.2的位置误差后，奖励随距离严格下降。旧位置V1奖励方向错误，且每个物理子步更新相对目标，不能作为本版本训练或评估依据。旧力矩V1及位置V1权重均被拒绝，必须从头训练。

此版本仍使用原有成功终止、5秒上限和PD增益；增量裁剪不是实际关节速度硬限幅，也未新增连续保持验收或硬件伺服标定。

## 重力补偿单变量实验

任务 `RebotArm-Reach-GravityComp-Mjlab`，契约 `reach-position-gravity-compensated-v3`。
以官方对齐版为基线，姿态奖励仍为-0.1，动作/PPO/目标采样不变。
仅编译模型增加原生body gravcomp（包括固定相机负载），六个机械臂关节
使用actuatorgravcomp，将补偿计入总关节驱动力，并在合并PD力矩后限制为
前三轴±27Nm、后三轴±7Nm。重力仍启用，XML资源和哈希不变。
这是仿真模型计算的理想重力前馈，不是已经标定的实机电机位置模式。
新旧任务权重不兼容，所有历史任务保持原有动力学及契约。

启动：`python scripts/train.py --experiment reach_gravity_aligned`，短验证
用`reach_gravity_aligned_smoke`。严格评估需新任务名，仍使用2–6cm目标、
100回合、250步、seed20000及连续25样本指标以便对照。


## 固定运动惩罚消融实验

任务`RebotArm-Reach-GravityComp-FixedPenalties-Mjlab`，契约
`reach-position-gravity-compensated-v3-fixed-penalties-v1`。
只取消重力补偿版的两项奖励课程：动作变化与关节速度惩罚全程保持
-0.0001，不再于4501控制步后增强。姿态奖励-0.1、PPO、动作、观测、
动力学、目标分布及时序均沿用重力补偿版。
按项目的实验契约隔离规则，新旧任务权重禁止交叉加载；虽然推理动作与
观测相同，这仍是独立训练配方，正式对照从头训练，不从旧权重恢复。

启动：`python scripts/train.py --experiment reach_gravity_fixed`；
短验证使用`reach_gravity_fixed_smoke`。128环境、24步、1000轮、seed7、
每50轮保存，与原重力补偿实验相同。使用相同固定目标分别比较中间与
最终权重，不能只用训练总奖励判断改善，因为奖励课程已不同。

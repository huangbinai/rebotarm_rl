# 迁移来源记录

源仓库：https://github.com/huangbinai/robotarm_ros2

源提交：00065fb9e47730a6905af5d6268082f8c6d02db2

原子目录：rebotarm_rl/；通过`git subtree split --prefix=rebotarm_rl`提取，保留子树历史。子树创建前的历史仍可在源仓库查询。

model_manifest.json固定源提交及所需XML/STL文件哈希。模型缓存独立于ROS工作区，本仓库不维护手工复制的另一份模型源码。

迁移时保留ROS原环境位置以兼容启动命令，RL新建独立.venv且不提交Git。后续ROS环境精简另行实施，不改变本记录的迁移事实。

## 首轮迁移检查

- 8项契约和资源测试通过。
- 从GitHub获取固定提交模型，14个文件哈希全部一致。
- 构建wheel并在无ROS依赖的临时环境安装，在两个仓库之外成功定位资源。
- 使用原运行环境完成8环境、1迭代GPU预检查，以及2回合、20步配对评估；当时独立环境依赖尚在下载，此项不算独立环境验收。
- 未操作实机或远程云实例。

原本地runs/mjlab及runs/rl-package-smoke复制到runs/imported-robotarm-ros2/。ROS目录保留原输出，以免破坏历史证据路径；两份均被Git忽略。

## 独立环境验收（2026-10-04）

- 新.venv使用Python 3.12、torch 2.14.1+cu130、MuJoCo 3.11.0和RTX 4060 Laptop GPU，无rclpy或rebotarm_simulation。
- 从/tmp执行并清除PYTHONPATH、AMENT_PREFIX_PATH：任务注册通过；V1完成8环境、1次PPO迭代并生成model_0.pt。
- 新环境配对评估完成2回合、20步，输出有限；CPU和GPU最终成功次数均为0。这是链路验证，不代表策略训练完成或收敛。
- 8项独立契约/资源测试和依赖检查通过。
- 初始pytest受ROS继承的PYTHONPATH影响并自动加载ROS插件；清理后测试通过，README记录了清理方式。
- mjlab 1.6.0未声明cu130可选依赖，已移除被忽略的依赖选项，保留显式固定的torch CUDA版本。
- ROS移除训练代码后：全套757通过、37跳过；分层25通过；物理测试96通过；仿真包colcon重建和编译检查通过。全套数量变化来自训练测试迁出。
- 原ROS训练插件已卸载，当时未移除MuJoCo依赖。
- 均为本地验证；未声称执行GitHub CI、云任务或Isaac Lab任务。

详细日志和权重不进入Git。迁移环境版本保存在requirements/migration-snapshot.txt。后续结构改造和V2验证见docs/decisions目录。

"""Reach专用回放显示；标记仅写入渲染场景，不参与物理仿真。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import threading

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer
from mjlab.viewer.viewer_config import ViewerConfig

from rebotarm_rl.contracts.policy import REACH_GRAVITY_FIXED
from rebotarm_rl.training.rsl_rl.reach_ppo import aligned_runner_cfg
from .runner import RecordedRunner
from .tasks.reach.aligned import make_gravity_fixed_env_cfg


def playback_config(target_mode: str, seed: int):
    """创建独立回放配置；固定模式仅在手动reset时换目标，无自动重置。"""
    if target_mode not in ("fixed", "refresh") or seed < 0:
        raise ValueError("target_mode须为fixed/refresh，seed须为非负整数")
    cfg = make_gravity_fixed_env_cfg(play=True, num_envs=1)
    cfg.seed = seed
    cfg.terminations = {}  # 避免12秒超时重置被误认为追踪抖动。
    if target_mode == "fixed":
        cfg.commands["reach"].resampling_time_range = (1e9, 1e9)
    cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        lookat=(0.15, 0.0, 0.20), distance=0.9, azimuth=135, elevation=-22,
    )
    return cfg


def reach_state(base):
    """读取当前世界系TCP/目标位姿(wxyz)与实时误差，不使用reset缓存指标。"""
    command = base.command_manager.get_term("reach")
    current = command.robot.data.site_pose_w[0, command.site_id].detach().cpu().numpy()
    target = command.command[0].detach().cpu().numpy()
    position_error = float(np.linalg.norm(current[:3] - target[:3]))
    a, b = current[3:].astype(np.float64), target[3:].astype(np.float64)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    # 归一化四元数的最短弧夹角；q与-q等价，小角度时避免acos精度损失。
    orientation_error = float(4 * np.arctan2(min(np.linalg.norm(a-b), np.linalg.norm(a+b)),
                                            max(np.linalg.norm(a-b), np.linalg.norm(a+b))))
    success = (position_error < REACH_GRAVITY_FIXED.success_position_m
               and orientation_error < REACH_GRAVITY_FIXED.success_orientation_rad)
    return current, target, position_error, orientation_error, success


def draw_reach(visualizer, current, target) -> None:
    """目标：黄球、长RGB轴；TCP：青球、短浅色轴；灰色网格仅用于显示。"""
    for value in np.linspace(-0.4, 0.6, 11):
        visualizer.add_cylinder(np.array([value, -0.4, 0.]), np.array([value, 0.6, 0.]),
                                0.0005, (0.3, 0.3, 0.3, 1.))
        visualizer.add_cylinder(np.array([-0.4, value, 0.]), np.array([0.6, value, 0.]),
                                0.0005, (0.3, 0.3, 0.3, 1.))
    for pose, scale, color, axes in (
        (target, .065, (1., .8, .0, 1.), None),
        (current, .035, (0., 1., 1., 1.), ((1., .5, .5), (.5, 1., .5), (.5, .5, 1.))),
    ):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, pose[3:].astype(np.float64))
        visualizer.add_frame(pose[:3], rotation.reshape(3, 3), scale=scale,
                             axis_radius=.0015, axis_colors=axes)
        visualizer.add_sphere(pose[:3], .004, color)
    if np.linalg.norm(current[:3] - target[:3]) > 1e-6:
        visualizer.add_cylinder(current[:3], target[:3], .0007, (1., .8, .0, 1.))


class ReachViewer(NativeMujocoViewer):
    """适配已固定版本mjlab的native显示钩子；沿用原生暂停、速度及reset控制。"""
    def __init__(self, env, policy, target_mode):
        super().__init__(env, policy, enable_perturbations=False)
        self.target_mode = target_mode
        self._render_thread = None

    def setup(self):
        existing = set(threading.enumerate())
        super().setup()
        # 固定MuJoCo版本的launch_passive使用daemon线程且Handle.close仅发出退出请求。
        # 仅跟踪本窗口新建的绘图线程，避免Python退出时GLFW销毁与绘图清理竞争。
        for thread in set(threading.enumerate()) - existing:
            if getattr(thread, "_target", None) is mujoco.viewer._launch_internal:
                self._render_thread = thread

    def close(self):
        super().close()
        if self._render_thread is not None:
            self._render_thread.join(timeout=5.)
            if self._render_thread.is_alive():
                raise RuntimeError("MuJoCo绘图线程未正常退出，请检查桌面OpenGL驱动")
            self._render_thread = None

    def _update_debug_visualizers(self, viewer):
        super()._update_debug_visualizers(viewer)
        if self._show_debug_vis:
            visualizer = MujocoNativeDebugVisualizer(viewer.user_scn, self.mjm, 0)
            draw_reach(visualizer, *reach_state(self.env.unwrapped)[:2])

    def _set_status_overlay(self, viewer):
        current, target, pe, oe, success = reach_state(self.env.unwrapped)
        status = self.get_status()
        labels = "Mode\nPlayback\nPosition error\nOrientation error\nInstant success\nTarget XYZ (m)\nTCP XYZ (m)\nMarkers\nControls"
        values = (
            f"{self.target_mode} (no auto reset)\n"
            f"{'PAUSED' if status.paused else 'RUNNING'} | {status.speed_label}\n"
            f"{pe * 1000:.2f} mm (<10 mm)\n{np.degrees(oe):.2f} deg (<3 deg)\n"
            f"{'YES' if success else 'NO'} (not hold-success)\n"
            f"{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}\n"
            f"{current[0]:.3f}, {current[1]:.3f}, {current[2]:.3f}\n"
            "Target: yellow / long RGB; TCP: cyan / short axes\n"
            "Space: pause | Enter: reset/new target | R: markers | +/-: speed"
        )
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
                          mujoco.mjtGridPos.mjGRID_TOPLEFT.value, labels, values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-mode", choices=("fixed", "refresh"), default="fixed",
                        help="fixed:目标保持至Enter重置；refresh:每4秒仿真时间换目标")
    parser.add_argument("--seed", type=int, default=7, help="回放目标种子，不是训练种子")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed必须非负")
    checkpoint = args.checkpoint.resolve(strict=True)
    agent_cfg = aligned_runner_cfg()
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(playback_config(args.target_mode, args.seed),
                                              device="cuda:0"), clip_actions=agent_cfg.clip_actions)
    try:
        runner = RecordedRunner(env, asdict(agent_cfg), device="cuda:0")
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        print(f"Reach回放模式：{args.target_mode}；仅仿真推理，不训练。目标采样0–6cm，非独立评估。")
        ReachViewer(env, policy, args.target_mode).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()

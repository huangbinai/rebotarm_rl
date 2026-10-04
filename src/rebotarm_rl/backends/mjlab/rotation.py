"""wxyz 四元数的相对旋转计算，供 GPU 观测和 CPU 配对评估共用。"""
import torch


def rotation_error_wxyz(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """返回世界系 current * inverse(target) 的最短旋转向量，单位 rad。

    输入为有限非零四元数，形状 (..., 4)，顺序 wxyz；输出 (..., 3)。
    q 与 -q 表示同一旋转。恰好 π 时选择绝对值最大轴分量为正，
    使双覆盖输入获得相同结果；π 附近仍有 SO(3) 对数映射固有的不连续。
    """
    if current.ndim < 1 or current.shape != target.shape or current.shape[-1] != 4:
        raise ValueError("四元数形状必须相同且末维为4")
    # 观测来源是物理引擎的单位四元数，归一化消除积分与浮点误差。
    current = current / current.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cw, cv = current[..., :1], current[..., 1:]
    tw, tv = target[..., :1], target[..., 1:]
    w = cw * tw + (cv * tv).sum(dim=-1, keepdim=True)
    xyz = tw * cv - cw * tv - torch.linalg.cross(cv, tv, dim=-1)
    # w>=0 将角度约束到[0,π]；w==0 时再固定轴方向。
    pivot = xyz.gather(-1, xyz.abs().argmax(dim=-1, keepdim=True))
    flip = (w < 0) | ((w == 0) & (pivot < 0))
    sign = torch.where(flip, -torch.ones_like(w), torch.ones_like(w))
    w, xyz = w * sign, xyz * sign
    sine = xyz.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sine, w)
    # 接近零时 angle/sin(angle/2) -> 2，避免0/0和微小转角精度损失。
    scale = torch.where(sine > 1e-7, angle / sine.clamp_min(1e-12),
                        2 + sine.square() / 3)
    return xyz * scale

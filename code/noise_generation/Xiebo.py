"""Original generator functions; interactive examples omitted."""
import numpy as np

import matplotlib.pyplot as plt

def generate_transient_oscillation_noise(
    n,
    fs=150,
    mode="random",
    event_range=(1, 5),
    amp_range=(1.0, 5.0),
    width_ratio_range=(0.04, 0.12),
    cycles_range=(3, 12),
    decay_range=(3.0, 10.0),
    sigma_range=(0.10, 0.25),
    min_gap=None,
    allow_overlap=False,
    add_center_spike=True,
    center_spike_prob=0.5,
    return_meta=False,
    seed=None
):
    """
    生成 AMT/MT 瞬态振荡类噪声。

    mode:
        "damped" : 衰减振荡，类似暂态响应后的振铃
        "packet" : 高斯包络振荡包，更像局部谐波包
        "random" : damped 和 packet 随机混合

    n:
        噪声长度，例如 100、200、300、500、1500、5000。

    width_ratio_range:
        每段瞬态振荡持续长度占总长度 n 的比例。

    cycles_range:
        每段振荡内部包含的周期数。
        比直接写 Hz 更稳定，适合不同长度噪声库。

    decay_range:
        damped 模式下的衰减强度，越大衰减越快。

    sigma_range:
        packet 模式下高斯包络宽度，越小越集中。

    min_gap:
        不同振荡事件之间的最小间隔。
        若为 None，则自动取 0.03*n。

    add_center_spike:
        是否在局部振荡包中加入一个中心强尖峰，使其更像实测暂态干扰。

    return_meta:
        是否返回事件参数。
    """

    if seed is not None:
        np.random.seed(seed)

    if mode not in ["damped", "packet", "random"]:
        raise ValueError("mode 必须是 'damped', 'packet', 或 'random'")

    noise = np.zeros(n)
    occupied = []
    meta = []

    if min_gap is None:
        min_gap = max(5, int(0.03 * n))

    event_num = np.random.randint(event_range[0], event_range[1] + 1)

    for event_id in range(event_num):

        if mode == "random":
            current_mode = np.random.choice(["damped", "packet"])
        else:
            current_mode = mode

        # 当前事件宽度
        width_low = max(8, int(round(width_ratio_range[0] * n)))
        width_high = max(width_low + 1, int(round(width_ratio_range[1] * n)))
        width_high = min(width_high, n - 1)

        if width_high <= width_low:
            width_high = min(n - 1, width_low + 1)

        width = np.random.randint(width_low, width_high + 1)

        if width <= 3 or width >= n:
            continue

        A = np.random.uniform(*amp_range) * np.random.choice([-1, 1])

        u = np.linspace(0, 1, width)
        cycles = np.random.uniform(*cycles_range)
        phase = np.random.uniform(0, 2 * np.pi)

        # 轻微频率漂移，让振荡不完全理想
        chirp_strength = np.random.uniform(-0.8, 0.8)
        phase_term = 2 * np.pi * (cycles * u + chirp_strength * (u - 0.5) ** 2) + phase
        carrier = np.sin(phase_term)

        # ------------------------------------------------------------
        # 模式1：衰减振荡
        # ------------------------------------------------------------
        if current_mode == "damped":
            decay = np.random.uniform(*decay_range)

            # attack 让波形不是从最大值突兀开始
            attack = np.random.uniform(10, 30)
            envelope = (1 - np.exp(-attack * u)) * np.exp(-decay * u)

            seg = A * envelope * carrier

            sigma = None
            center = None

        # ------------------------------------------------------------
        # 模式2：高斯包络振荡包
        # ------------------------------------------------------------
        elif current_mode == "packet":
            center = np.random.uniform(0.35, 0.65)
            sigma = np.random.uniform(*sigma_range)

            envelope = np.exp(-0.5 * ((u - center) / sigma) ** 2)

            seg = A * envelope * carrier

            decay = None

        else:
            raise ValueError("未知 current_mode")

        # ------------------------------------------------------------
        # 可选：加入中心强尖峰，模拟实测暂态振荡中的突发高幅值
        # ------------------------------------------------------------
        if add_center_spike and np.random.rand() < center_spike_prob:
            spike_idx = np.argmax(np.abs(seg))
            spike_amp = np.random.uniform(0.5, 1.5) * A

            # 1点或2点尖峰
            spike_width = np.random.choice([1, 1, 2])

            end_idx = min(width, spike_idx + spike_width)
            seg[spike_idx:end_idx] += spike_amp

        # ------------------------------------------------------------
        # 找位置
        # ------------------------------------------------------------
        placed = False

        for _ in range(200):
            start = np.random.randint(0, n - width + 1)
            end = start + width

            if allow_overlap:
                placed = True
                break

            valid = True
            for s, e in occupied:
                if not (end + min_gap <= s or start - min_gap >= e):
                    valid = False
                    break

            if valid:
                occupied.append((start, end))
                placed = True
                break

        if not placed:
            continue

        noise[start:end] += seg

        meta.append({
            "event_id": event_id + 1,
            "mode": current_mode,
            "start": start,
            "end": end,
            "width": width,
            "amplitude": A,
            "cycles": cycles,
            "decay": decay,
            "center": center,
            "sigma": sigma
        })

    if return_meta:
        return noise, meta
    else:
        return noise

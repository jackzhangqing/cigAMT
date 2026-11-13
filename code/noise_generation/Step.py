"""Original generator functions; interactive examples omitted."""
import numpy as np

import matplotlib.pyplot as plt

def rand_range(a, b):
    return a + (b - a) * np.random.rand()

def generate_step_noise(
    n,
    fs=150,
    mode="matlab",
    A=3.0,
    amp_range=(0.5, 3.0),
    jump_range=None,
    max_jumps=5,
    drift=True,
    drift_ratio=0.3,
    margin_ratio=0.05,
    return_meta=False,
    seed=None
):
    """
    生成 AMT/MT 阶跃跳变 / 基线突变噪声。

    mode:
        "matlab"   : 对应 case 'step'
                     随机阶跃点 + offset 分段保持 + 低频漂移
        "dendrite" : 对应 step_jump_dendrite2025
                     避开边缘，在中间区域随机生成阶跃点
        "random"   : 随机选择 matlab 或 dendrite

    n:
        噪声长度，例如 100、200、300、500、1500。

    A:
        matlab 模式下的基础幅值，对应原代码中的 A。

    amp_range:
        dendrite 模式下的阶跃幅值范围。

    jump_range:
        阶跃次数范围。若为 None，则自动按照 n 设置。

    max_jumps:
        最大阶跃次数，默认 5。

    drift:
        是否叠加轻微低频漂移。matlab 模式默认建议 True。

    drift_ratio:
        漂移幅值上限比例，对应原代码 driftAmp = rand_range(0, 0.3*A)。

    margin_ratio:
        dendrite 模式下避开首尾边缘的比例。

    return_meta:
        是否返回参数信息。
    """

    if seed is not None:
        np.random.seed(seed)

    if mode not in ["matlab", "dendrite", "random"]:
        raise ValueError("mode 必须是 'matlab', 'dendrite', 或 'random'")

    if mode == "random":
        current_mode = np.random.choice(["matlab", "dendrite"])
    else:
        current_mode = mode

    y = np.zeros(n)
    meta = {
        "type": "step",
        "mode": current_mode,
        "n": n,
        "fs": fs,
        "jump_points": [],
        "levels": [],
        "drift_amp": 0.0,
        "A": A,
    }

    # ============================================================
    # 方式一：严格对应 MATLAB case 'step'
    # ============================================================
    if current_mode == "matlab":

        if jump_range is None:
            # jumpNum = randi([1, max(1, min(5, round(N/80)))])
            upper = max(1, min(max_jumps, round(n / 80)))
            jump_range = (1, upper)

        jump_num = np.random.randint(jump_range[0], jump_range[1] + 1)

        if n <= 2:
            if return_meta:
                return y, meta
            return y

        # cps = sort(randi([2, N-1], 1, jumpNum))
        # Python 下标从 0 开始，因此对应位置取 1 到 n-2
        cps = np.sort(np.random.randint(1, n - 1, size=jump_num))

        offset = 0.0
        last_idx = 0

        for cp in cps:
            y[last_idx:cp] = offset
            meta["levels"].append(offset)

            step_amp = np.sign(np.random.randn()) * A * rand_range(0.3, 1.2)
            if step_amp == 0:
                step_amp = A * rand_range(0.3, 1.2)

            offset = offset + step_amp
            last_idx = cp
            meta["jump_points"].append(int(cp))

        y[last_idx:] = offset
        meta["levels"].append(offset)

        # 叠加轻微低频漂移
        if drift:
            drift_amp = rand_range(0, drift_ratio * A)
            y = y + drift_amp * np.linspace(-1, 1, n)
            meta["drift_amp"] = drift_amp

    # ============================================================
    # 方式二：对应 step_jump_dendrite2025
    # ============================================================
    elif current_mode == "dendrite":

        if jump_range is None:
            # n_steps_range = (1, max(2, n // 300))
            jump_range = (1, max(2, n // 300))

        jump_num = np.random.randint(jump_range[0], jump_range[1] + 1)

        margin = max(20, int(n * margin_ratio))

        if n - 2 * margin <= jump_num:
            if return_meta:
                return y, meta
            return y

        possible_positions = np.arange(margin, n - margin)

        jump_num = min(jump_num, len(possible_positions))

        cps = np.sort(
            np.random.choice(possible_positions, size=jump_num, replace=False)
        )

        level = 0.0
        prev = 0

        for cp in cps:
            y[prev:cp] = level
            meta["levels"].append(level)

            step_amp = np.random.uniform(*amp_range) * np.random.choice([-1, 1])
            level = level + step_amp

            prev = cp
            meta["jump_points"].append(int(cp))

        y[prev:] = level
        meta["levels"].append(level)

    if return_meta:
        return y, meta
    else:
        return y

"""Original generator functions; interactive examples omitted."""
import numpy as np

import matplotlib.pyplot as plt

def generate_sharp_pulse_noise(
    n,
    fs=150,
    mode="sparse",
    pulse_range=None,
    amp_range=(1.0, 5.0),
    W_range=(1, 4),
    single_point_prob=0.7,
    min_gap=20,
    allow_overlap=False,
    return_meta=False,
    seed=None
):
    """
    生成 AMT/MT 尖脉冲/尖峰噪声。

    mode:
        "single" : 只生成少量孤立尖脉冲
        "sparse" : 稀疏尖脉冲，推荐用于 AMT spike
        "random" : 随机生成孤立尖脉冲，但仍保持稀疏分布

    n:
        噪声长度，例如 100, 200, 300, 500, 1500。

    pulse_range:
        脉冲数量范围。
        若为 None，则根据 n 自动设置较稀疏的数量。

    amp_range:
        脉冲幅值范围。

    W_range:
        脉冲宽度范围。建议保持在 (1, 4)，不要太宽。

    single_point_prob:
        单点脉冲概率。默认 0.7，表示 70% 的脉冲宽度为 1。

    min_gap:
        脉冲之间最小间隔，避免多个脉冲挤在一起形成簇状或短方波。

    allow_overlap:
        是否允许脉冲重叠。尖脉冲类一般建议 False。

    return_meta:
        True 时返回噪声和脉冲参数记录。

    seed:
        随机种子。
    """

    if seed is not None:
        np.random.seed(seed)

    if mode not in ["single", "sparse", "random"]:
        raise ValueError("mode 必须是 'single', 'sparse', 或 'random'")

    noise = np.zeros(n)
    occupied = []
    meta = []

    # ------------------------------------------------------------
    # 1. 决定脉冲数量
    # ------------------------------------------------------------
    if pulse_range is None:
        if mode == "single":
            pulse_range = (1, max(2, n // 300))
        else:
            pulse_range = (
                max(2, n // 250),
                max(4, n // 100)
            )

    pulse_num = np.random.randint(pulse_range[0], pulse_range[1] + 1)

    # ------------------------------------------------------------
    # 2. 逐个生成尖脉冲
    # ------------------------------------------------------------
    for pulse_id in range(pulse_num):

        A = np.random.uniform(*amp_range) * np.random.choice([-1, 1])

        # 70% 概率为单点尖脉冲，30% 概率为 2–4 点窄脉冲
        if np.random.rand() < single_point_prob:
            W = 1
        else:
            W = np.random.randint(max(2, W_range[0]), W_range[1] + 1)

        if W >= n:
            W = 1

        # ------------------------------------------------------------
        # 3. 寻找满足间隔约束的位置
        # ------------------------------------------------------------
        placed = False

        for _ in range(100):
            t0 = np.random.randint(0, n - W + 1)
            t1 = t0 + W

            if allow_overlap:
                placed = True
                break

            valid = True

            for s, e in occupied:
                # 保证两个脉冲之间至少相隔 min_gap
                if not (t1 + min_gap <= s or t0 - min_gap >= e):
                    valid = False
                    break

            if valid:
                occupied.append((t0, t1))
                placed = True
                break

        if not placed:
            continue

        noise[t0:t1] = A

        meta.append({
            "pulse_id": pulse_id + 1,
            "start": t0,
            "end": t1,
            "width": W,
            "amplitude": A
        })

    if return_meta:
        return noise, meta
    else:
        return noise

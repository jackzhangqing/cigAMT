"""Original generator functions; interactive examples omitted."""
import numpy as np

import matplotlib.pyplot as plt

def generate_tri_charge_noise(
    n,
    fs=150,
    mode="random",
    event_range=(2, 6),
    amp_range=(0.5, 3.0),
    W1_range=(5, 40),
    W2_range=(10, 60),
    rc1_range=(2, 5),  
    jj_range=(5, 20),
    allow_overlap=False,
    ensure_both=True,
    return_meta=False,
    seed=None
):
    """
    统一生成 AMT/MT 充放电三角波噪声。

    参数说明
    ----------
    n : int
        噪声长度，例如 100, 200, 300, 500, 1500。

    fs : int or float
        采样率，默认 150 Hz。当前函数主要按采样点控制宽度，
        fs 保留用于后续扩展或记录。

    mode : str
        三角波构建方式：
        "linear" : 分段线性三角波
        "exp"    : 指数充放电型三角波
        "random" : 随机增强型，随机选择 linear 或 exp

    event_range : tuple
        每条噪声中三角波事件数量范围，例如 (1, 3)。

    amp_range : tuple
        幅值范围。

    width_ratio_range : tuple
        linear / random 模式中，事件宽度相对于 n 的比例范围。

    W1_range, W2_range : tuple
        linear 模式中，上升段和下降段的采样点宽度范围。
        当使用 width_ratio_range 时，函数会优先根据总 width 再划分 W1/W2。

    rc1_range : tuple
        exp 模式中，指数充电时间常数范围。

    jj_range : tuple
        exp 模式中，正负相位之间的间隔范围。

    allow_overlap : bool
        是否允许多个三角波事件重叠。
        False 表示尽量避免重叠。

    return_meta : bool
        True 时返回 noise 和每个事件的参数信息。

    seed : int or None
        随机种子，便于复现。

    返回
    ----------
    noise : ndarray
        长度为 n 的三角波噪声。

    meta : list of dict, optional
        每个事件的参数记录。
    """

    if seed is not None:
        np.random.seed(seed)

    if mode not in ["linear", "exp", "random"]:
        raise ValueError("mode 必须是 'linear', 'exp', 或 'random'")

    noise = np.zeros(n)
    occupied = []
    meta = []

    event_num = np.random.randint(event_range[0], event_range[1] + 1)  # 随机生成事件数量

    # random 模式下，强制至少包含一个 linear 和一个 exp
    # ============================================================
    if mode == "random":
        if ensure_both:
            event_num = max(event_num, 2)

            mode_list = ["linear", "exp"]

            if event_num > 2:
                extra_modes = list(
                    np.random.choice(["linear", "exp"], size=event_num - 2)
                )
                mode_list.extend(extra_modes)

            np.random.shuffle(mode_list)
        else:
            mode_list = list(
                np.random.choice(["linear", "exp"], size=event_num)
            )
    else:
        mode_list = [mode] * event_num

    

     # 3. 逐个事件生成并放入 noise
    # ============================================================
    for event_id, current_mode in enumerate(mode_list):

        A = None
        A1 = None
        A2 = None
        W1 = None
        W2 = None
        rc1 = None
        rc2 = None
        jj = None

        # ------------------------------------------------------------
        # 方式一：分段线性三角波
        # ------------------------------------------------------------
        if current_mode == "linear":

            A1 = np.random.uniform(*amp_range) * np.random.choice([-1, 1])
            A2 = np.random.uniform(*amp_range) * np.random.choice([-1, 1])

            W1 = np.random.randint(W1_range[0], W1_range[1] + 1)
            W2 = np.random.randint(W2_range[0], W2_range[1] + 1)

            width = W1 + W2

            if width >= n:
                width = n - 1
                W1 = max(2, width // 2)
                W2 = width - W1

            seg = np.zeros(width)

            for i in range(W1):
                if i < width:
                    seg[i] = A1 * (i / W1)

            for i in range(W2):
                idx = W1 + i
                if idx < width:
                    seg[idx] = A2 * (1 - i / W2)

        # ------------------------------------------------------------
        # 方式二：指数充放电型三角波
        # ------------------------------------------------------------
        elif current_mode == "exp":

            A = np.random.uniform(*amp_range) * np.random.choice([-1, 1])

            rc1 = np.random.randint(rc1_range[0], rc1_range[1] + 1)
            rc2 = rc1 * 3
            jj = np.random.randint(jj_range[0], jj_range[1] + 1)

            t1 = np.arange(1, rc1 * 5 + 1)
            charge = A * (1 - np.exp(-t1 / rc1))

            t2 = np.arange(1, rc2 * 5 + 1)
            discharge = A * np.exp(-t2 / rc2)

            positive = np.concatenate([charge, discharge])
            negative = -positive
            gap = np.zeros(jj)

            seg = np.concatenate([positive, gap, negative])

            width = len(seg)

            if width >= n:
                seg = seg[:n - 1]
                width = len(seg)

            W1 = rc1 * 5
            W2 = rc2 * 5

        else:
            raise ValueError("未知事件类型")

        if width <= 0 or n - width < 0:
            continue

        # ------------------------------------------------------------
        # 4. 寻找添加位置
        # ------------------------------------------------------------
        placed = False

        for _ in range(200):
            start = np.random.randint(0, n - width + 1)
            end = start + width

            if allow_overlap:
                placed = True
                break

            no_overlap = all(end <= s or start >= e for s, e in occupied)

            if no_overlap:
                occupied.append((start, end))
                placed = True
                break

        # 如果放不下，就跳过该事件
        if not placed:
            continue

        # ------------------------------------------------------------
        # 5. 叠加到噪声序列中
        # ------------------------------------------------------------
        noise[start:start + width] += seg[:width]

        meta.append({
            "event_id": event_id + 1,
            "mode": current_mode,
            "start": start,
            "end": start + width,
            "width": width,
            "A": A,
            "A1": A1,
            "A2": A2,
            "W1": W1,
            "W2": W2,
            "rc1": rc1,
            "rc2": rc2,
            "jj": jj
        })

    if return_meta:
        return noise, meta
    else:
        return noise

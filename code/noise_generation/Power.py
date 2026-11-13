"""Original generator functions; interactive examples omitted."""
import numpy as np

import matplotlib.pyplot as plt

def rand_range(a, b):
    return a + (b - a) * np.random.rand()

def powerline_noise(n, fs=150, A=3.0, seed=None):
    """
    完全按照 MATLAB case 'powerline' 逻辑构建工频干扰：
    50/60 Hz + 若干高次谐波 + ±0.05 Hz 频率漂移 + 轻微幅值调制
    """

    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n) / fs
    y = np.zeros(n)

    # ------------------------------------------------------------
    # 1. 随机选择 50 Hz 或 60 Hz
    # ------------------------------------------------------------
    if np.random.rand() < 0.5:
        f0 = 50.0
    else:
        f0 = 60.0

    # ------------------------------------------------------------
    # 2. 加入 ±0.05 Hz 频率漂移
    # ------------------------------------------------------------
    f0 = f0 + rand_range(-0.05, 0.05)

    # ------------------------------------------------------------
    # 3. 根据 Nyquist 频率计算最大谐波阶数
    # ------------------------------------------------------------
    maxH = int(np.floor((fs / 2) / f0))

    if maxH < 1:
        maxH = 1

    # ------------------------------------------------------------
    # 4. 随机选择实际谐波数量
    # harmonicNum = randi([1, max(1, min(5, maxH))])
    # ------------------------------------------------------------
    harmonicNum = np.random.randint(1, max(1, min(5, maxH)) + 1)

    # ------------------------------------------------------------
    # 5. 逐阶谐波叠加
    # ------------------------------------------------------------
    for h in range(1, harmonicNum + 1):

        fh = h * f0

        if fh >= fs / 2:
            continue

        phase = 2 * np.pi * np.random.rand()

        Ah = A / h * rand_range(0.5, 1.2)

        # 轻微幅值调制，让工频不完全理想
        fm = rand_range(0.05, 0.5)

        am = 1 + rand_range(0, 0.15) * np.sin(
            2 * np.pi * fm * t + 2 * np.pi * np.random.rand()
        )

        y = y + Ah * am * np.sin(2 * np.pi * fh * t + phase)

    return y

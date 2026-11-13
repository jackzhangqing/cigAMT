"""Original generator functions; interactive examples omitted."""
import numpy as np

import matplotlib.pyplot as plt

def rand_range(a, b):
    return a + (b - a) * np.random.rand()

def quasi_sine_noise(n, A=3.0, seed=None):
    """
    准正弦周期干扰：频率、幅值缓慢变化

    完全对应 MATLAB 逻辑：

        cycles = rand_range(2, max(3, N/20));
        phase = 2*pi*rand;

        u = linspace(0, 1, N);
        freqJitter = 0.03 * cumsum(randn(1, N));
        freqJitter = freqJitter / (max(abs(freqJitter)) + eps);

        ampMod = 1 + rand_range(0.1, 0.5) * 
                 sin(2*pi*rand_range(0.5, 3)*u + 2*pi*rand);

        y = A * ampMod .* sin(2*pi*cycles*u + phase + freqJitter);

    参数
    ----------
    n : int
        噪声长度，例如 100、200、300、500、1500。

    A : float
        基础幅值。

    seed : int or None
        随机种子，便于复现。

    返回
    ----------
    y : ndarray
        长度为 n 的准正弦周期干扰。
    """

    if seed is not None:
        np.random.seed(seed)

    N = n

    # 1. 周期数随机
    cycles = rand_range(2, max(3, N / 20))

    # 2. 初始相位随机
    phase = 2 * np.pi * np.random.rand()

    # 3. 归一化时间轴
    u = np.linspace(0, 1, N)

    # 4. 累积型频率扰动：让频率缓慢漂移
    freqJitter = 0.03 * np.cumsum(np.random.randn(N))
    freqJitter = freqJitter / (np.max(np.abs(freqJitter)) + np.finfo(float).eps)

    # 5. 幅值缓慢调制
    ampMod = 1 + rand_range(0.1, 0.5) * np.sin(
        2 * np.pi * rand_range(0.5, 3) * u + 2 * np.pi * np.random.rand()
    )

    # 6. 准正弦周期干扰
    y = A * ampMod * np.sin(
        2 * np.pi * cycles * u + phase + freqJitter
    )

    return y

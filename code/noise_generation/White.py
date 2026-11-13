"""Original generator functions; interactive examples omitted."""
import numpy as np

import matplotlib.pyplot as plt

def rand_range(a, b):
    return a + (b - a) * np.random.rand()

def gaussian_white_noise(
    n,
    fs=150,
    amp_range=(0.2, 5.0),
    use_random_amp=True,
    A=None,
    return_meta=False,
    seed=None
):
    """
    高斯白噪声 Gaussian White Noise

    对应两种逻辑：
    1. 基础版：
        y = randn(N)

    2. MATLAB case 'white'：
        A = 10^(rand_range(log10(0.2), log10(5.0)))
        y = A * randn(1, N)

    参数
    ----------
    n : int
        噪声长度，例如 100、200、300、500、1500。

    fs : float
        采样率。白噪声本身不直接使用 fs，但保留该参数，方便和其他噪声函数接口统一。

    amp_range : tuple
        随机幅值范围，默认 (0.2, 5.0)。

    use_random_amp : bool
        True  : 使用对数均匀随机幅值 A。
        False : 若 A=None，则生成标准高斯白噪声 randn(N)。

    A : float or None
        指定固定幅值。如果不为 None，则使用 y = A * randn(N)。

    return_meta : bool
        是否返回参数信息。

    seed : int or None
        随机种子，便于复现。
    """

    if seed is not None:
        np.random.seed(seed)

    # 1. 决定幅值 A
    if A is not None:
        amp = A
        amp_mode = "fixed"
    elif use_random_amp:
        amp = 10 ** rand_range(np.log10(amp_range[0]), np.log10(amp_range[1]))
        amp_mode = "log_uniform"
    else:
        amp = 1.0
        amp_mode = "standard"

    # 2. 生成高斯白噪声
    y = amp * np.random.randn(n)

    meta = {
        "type": "white",
        "n": n,
        "fs": fs,
        "amplitude": amp,
        "amp_mode": amp_mode,
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "rms": float(np.sqrt(np.mean(y ** 2)))
    }

    if return_meta:
        return y, meta
    else:
        return y

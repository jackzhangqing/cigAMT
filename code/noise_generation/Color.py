"""Original generator functions; interactive examples omitted."""
import numpy as np

import matplotlib.pyplot as plt

def pink_noise_1overf(
    n=None,
    fs=150,
    dur=None,
    A=1.0,
    normalize=True,
    return_meta=False,
    seed=None
):
    """
    生成 pink noise / 1/f 噪声。

    对应 MATLAB colored_noise(Sf, dur, b) 中 b=1 的情况：

        b = 1  ->  pink noise / 1/f noise

    参数
    ----------
    n : int or None
        噪声长度。如果给定 n，则直接生成长度为 n 的噪声。

    fs : float
        采样率。

    dur : float or None
        持续时间，单位秒。如果 n=None，则使用 n=round(fs*dur)。

    A : float
        幅值系数。若 normalize=True，则最终输出为 A * 标准化 pink noise。

    normalize : bool
        是否归一化到均值为 0，标准差为 1。

    return_meta : bool
        是否返回参数信息。

    seed : int or None
        随机种子。

    返回
    ----------
    y : ndarray
        pink noise / 1/f 噪声序列。
    """

    if seed is not None:
        np.random.seed(seed)

    # ------------------------------------------------------------
    # 1. 确定样本点数
    # ------------------------------------------------------------
    if n is None:
        if dur is None:
            raise ValueError("n 和 dur 至少需要提供一个")
        n = int(round(fs * dur))

    N = int(n)

    # MATLAB 里若 N 为奇数，则 M=N+1，保证频谱对称更方便
    if N % 2 == 1:
        M = N + 1
    else:
        M = N

    # ------------------------------------------------------------
    # 2. 生成白噪声并做 FFT
    # ------------------------------------------------------------
    x = np.random.rand(M)
    F = np.fft.fft(x)

    # ------------------------------------------------------------
    # 3. 构造频率轴
    # MATLAB:
    # xf = (0:M/2-1)*(Sf/M)
    # ------------------------------------------------------------
    half = M // 2
    xf = np.arange(0, half) * (fs / M)

    # ------------------------------------------------------------
    # 4. 1/f^b 功率律，这里 b=1
    # MATLAB:
    # p = 1./xf(2:end).^b
    # 注意跳过 0 Hz，避免除以 0
    # ------------------------------------------------------------
    b = 1.0
    p = 1.0 / (xf[1:] ** b)

    # ------------------------------------------------------------
    # 5. 调整正频率部分幅值，保留原相位
    # MATLAB:
    # R = sqrt((abs(F(2:length(xf))).^2).*p')
    # P = angle(F(2:length(xf)))
    # ------------------------------------------------------------
    R = np.sqrt((np.abs(F[1:half]) ** 2) * p)
    P = np.angle(F[1:half])

    F[1:half] = R * (np.cos(P) + 1j * np.sin(P))

    # ------------------------------------------------------------
    # 6. 构造共轭对称频谱
    # ------------------------------------------------------------
    F[half + 1:] = np.conj(F[1:half][::-1])

    # Nyquist 点保持实数，避免数值误差
    F[half] = np.real(F[half])

    # ------------------------------------------------------------
    # 7. IFFT 回到时域
    # ------------------------------------------------------------
    y = np.real(np.fft.ifft(F))

    # ------------------------------------------------------------
    # 8. 截取原始长度
    # ------------------------------------------------------------
    y = y[:N]

    # ------------------------------------------------------------
    # 9. 均值为 0，标准差为 1
    # ------------------------------------------------------------
    if normalize:
        y = y - np.mean(y)
        yrms = np.sqrt(np.mean(y ** 2))
        if yrms > 0:
            y = y / yrms

    # ------------------------------------------------------------
    # 10. 幅值缩放
    # ------------------------------------------------------------
    y = A * y

    meta = {
        "type": "pink_noise_1overf",
        "b": b,
        "n": N,
        "fs": fs,
        "duration": N / fs,
        "amplitude": A,
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "rms": float(np.sqrt(np.mean(y ** 2)))
    }

    if return_meta:
        return y, meta
    else:
        return y

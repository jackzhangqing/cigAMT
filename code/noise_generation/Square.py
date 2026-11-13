"""Original generator functions; interactive examples omitted."""
import numpy as np

import matplotlib.pyplot as plt

def rand_range(a, b):
    return a + (b - a) * np.random.rand()

def generate_square_wave_noise(
    n,
    fs=150,
    mode="random",
    amp_range=(0.5, 3.0),
    duty_range=(0.15, 0.75),
    period_ratio_range=(0.05, 0.35),
    freq_range=(0.3, 5.0),
    block_range=(1, 3),
    block_width_ratio_range=(0.1, 0.8),
    bipolar_prob=0.5,
    return_meta=False,
    seed=None
):
    """
    生成 AMT/MT 方波 / 矩形脉冲串噪声。

    mode:
        "period" : 按采样点周期构建，对应 MATLAB case 'square'
        "freq"   : 按频率构建，对应 square_wave_dendrite2025
        "random" : 随机选择 period 或 freq

    n:
        噪声长度，例如 100、200、300、500、1500。

    fs:
        采样率。mode="freq" 时用于把频率转换为周期点数。

    amp_range:
        方波幅值范围。

    duty_range:
        占空比范围。

    period_ratio_range:
        mode="period" 时，周期相对于 n 的比例范围。

    freq_range:
        mode="freq" 时，方波频率范围。

    block_range:
        mode="period" 时，随机保留几段局部方波。

    block_width_ratio_range:
        mode="period" 时，每段局部方波持续长度占 n 的比例。

    bipolar_prob:
        双极性方波概率。
        若为双极性：raw = +1 / -1；
        若为单极性：raw = 1 / 0。

    return_meta:
        是否返回参数信息。

    seed:
        随机种子。
    """

    if seed is not None:
        np.random.seed(seed)

    if mode not in ["period", "freq", "random"]:
        raise ValueError("mode 必须是 'period', 'freq', 或 'random'")

    if mode == "random":
        current_mode = np.random.choice(["period", "freq"])
    else:
        current_mode = mode

    A = np.random.uniform(*amp_range)
    duty = rand_range(*duty_range)
    y = np.zeros(n)

    meta = {
        "type": "square",
        "mode": current_mode,
        "n": n,
        "fs": fs,
        "amplitude": A,
        "duty": duty,
        "bipolar": None,
        "period": None,
        "frequency": None,
        "blocks": []
    }

    # ============================================================
    # 方式一：按采样点周期构建，对应 MATLAB case 'square'
    # ============================================================
    if current_mode == "period":

        period_low = max(8, int(round(period_ratio_range[0] * n)))
        period_high = max(10, int(round(period_ratio_range[1] * n)))

        if period_high <= period_low:
            period_high = period_low + 1

        period = np.random.randint(period_low, period_high + 1)
        phase_shift = np.random.randint(0, period)

        idx = np.arange(n)

        raw = ((idx + phase_shift) % period < duty * period).astype(float)

        # 有些方波是双极性，有些是单极性平台抬升
        bipolar = np.random.rand() < bipolar_prob

        if bipolar:
            raw = 2 * raw - 1

        # 随机只保留一段或多段，模拟局部方波干扰
        mask = np.zeros(n)

        block_num = np.random.randint(block_range[0], block_range[1] + 1)

        for _ in range(block_num):
            width_low = max(10, int(round(block_width_ratio_range[0] * n)))
            width_high = max(12, int(round(block_width_ratio_range[1] * n)))

            width_high = min(width_high, n)

            if width_high <= width_low:
                width_high = min(n, width_low + 1)

            width = np.random.randint(width_low, width_high + 1)

            if width >= n:
                start = 0
                width = n
            else:
                start = np.random.randint(0, n - width + 1)

            end = start + width
            mask[start:end] = 1

            meta["blocks"].append({
                "start": start,
                "end": end,
                "width": width
            })

        y = A * raw * mask

        meta["period"] = period
        meta["phase_shift"] = phase_shift
        meta["bipolar"] = bipolar
        meta["frequency"] = fs / period

    # ============================================================
    # 方式二：按频率构建，对应 square_wave_dendrite2025
    # ============================================================
    elif current_mode == "freq":

        freq = np.random.uniform(*freq_range)
        period = max(int(fs / freq), 2)

        raw = np.zeros(n)

        # 方波只在中间某段出现
        start = np.random.randint(0, max(1, n // 4))

        end_low = min(n - 1, 3 * n // 4)
        end = np.random.randint(end_low, n)

        bipolar = True

        for i in range(start, end):
            if (i - start) % period < period * duty:
                raw[i] = A
            else:
                raw[i] = -A

        y = raw

        meta["period"] = period
        meta["frequency"] = freq
        meta["bipolar"] = bipolar
        meta["blocks"].append({
            "start": start,
            "end": end,
            "width": end - start
        })

    if return_meta:
        return y, meta
    else:
        return y

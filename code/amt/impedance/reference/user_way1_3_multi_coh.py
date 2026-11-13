"""Selected author-supplied Welch-Huber numerical functions.

Extracted without modifying the functions. Original record-loading examples and
machine-specific paths are omitted. This is used only through impedance_welch.
"""
import numpy as np
from scipy.signal import get_window
MU0 = 4*np.pi*1e-7

def robust_complex_mean(x, method="mean", c=1.5, max_iter=20):
    """
    对一组复数做稳健平均。

    method:
        "mean"   : 普通平均
        "median" : 实部、虚部分别取中位数
        "huber"  : Huber M-估计，降低异常段权重
    """
    x = np.asarray(x)
    x = x[np.isfinite(x.real) & np.isfinite(x.imag)]

    if x.size == 0:
        return np.nan + 1j * np.nan

    if method == "mean":
        return np.mean(x)

    if method == "median":
        return np.median(x.real) + 1j * np.median(x.imag)

    if method == "huber":
        # 初值：复数中位数
        m = np.median(x.real) + 1j * np.median(x.imag)

        for _ in range(max_iter):
            r = np.abs(x - m)
            scale = np.median(r) / 0.6745

            if scale < 1e-20:
                break

            w = np.ones_like(r)
            idx = r > c * scale
            w[idx] = c * scale / np.maximum(r[idx], 1e-30)

            m_new = np.sum(w * x) / np.sum(w)

            if np.abs(m_new - m) < 1e-12:
                break

            m = m_new

        return m

    raise ValueError("method must be 'mean', 'median', or 'huber'")

def _Z_from_segments(SER_segs, SHR_segs, robust="mean"):
    """
    通用阻抗求解：

        Z = <E·R*> · pinv(<H·R*>)

    单站法：
        R = H
        SER = <E·H*>
        SHR = <H·H*>
        Z = <E·H*> · pinv(<H·H*>)

    远参考法：
        R = Hr
        SER = <E·Hr*>
        SHR = <H·Hr*>
        Z = <E·Hr*> · pinv(<H·Hr*>)

    输入：
        SER_segs: (nseg, 2, 2)，每段的 E·R* 矩阵
        SHR_segs: (nseg, 2, 2)，每段的 H·R* 矩阵

    返回：
        Z   : (2, 2) complex，阻抗张量
        SER : (2, 2) complex，平均后的 E-R 互谱
        SHR : (2, 2) complex，平均后的 H-R 互谱
    """
    SER = np.zeros((2, 2), dtype=complex)
    SHR = np.zeros((2, 2), dtype=complex)

    for a in range(2):
        for b in range(2):
            SER[a, b] = robust_complex_mean(SER_segs[:, a, b], method=robust)
            SHR[a, b] = robust_complex_mean(SHR_segs[:, a, b], method=robust)

    Z = SER @ np.linalg.pinv(SHR)
    return Z, SER, SHR

def multiple_coherence(E_segs, H_segs):
    """
    多重相干度 multiple coherence。

    E_segs: (nseg, 2)，每段的 [Ex, Ey]
    H_segs: (nseg, 2)，每段的 [Hx, Hy]

    返回：
        coh_ex: Ex 被 Hx、Hy 共同解释的程度
        coh_ey: Ey 被 Hx、Hy 共同解释的程度

    公式近似：
        gamma_Ex^2 = S_ExH · pinv(S_HH) · S_HEx / S_ExEx
        gamma_Ey^2 = S_EyH · pinv(S_HH) · S_HEy / S_EyEy
    """
    E_segs = np.asarray(E_segs)
    H_segs = np.asarray(H_segs)

    nseg = E_segs.shape[0]
    if nseg < 4:
        return np.nan, np.nan

    # <E E*>，这里只需要 Ex、Ey 各自的自功率
    SEE = np.mean(np.abs(E_segs) ** 2, axis=0)  # [S_ExEx, S_EyEy]

    # <E H*>
    SEH = np.einsum("na,nb->ab", E_segs, np.conj(H_segs)) / nseg

    # <H H*>
    SHH = np.einsum("na,nb->ab", H_segs, np.conj(H_segs)) / nseg
    SHH_inv = np.linalg.pinv(SHH)

    coh = np.zeros(2, dtype=float)

    for i in range(2):
        s_eh = SEH[i:i + 1, :]      # 1×2
        s_he = np.conj(s_eh.T)      # 2×1
        pred = (s_eh @ SHH_inv @ s_he).real.item()
        coh[i] = pred / max(SEE[i], 1e-30)

    coh = np.clip(coh, 0.0, 1.0)
    return coh[0], coh[1]

def _err_jackknife(SER_segs, SHR_segs, robust="mean"):
    """
    Jackknife 删一刀切误差估计。

    每次去掉一个分段，用剩余分段重新估计 Z，
    根据多次 Z 的离散程度估计 sigma_Z。
    """
    n = SER_segs.shape[0]
    if n < 4:
        return np.full((2, 2), np.nan)

    Z_jk = np.empty((n, 2, 2), dtype=complex)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Z_i, _, _ = _Z_from_segments(SER_segs[mask], SHR_segs[mask], robust=robust)
        Z_jk[i] = Z_i

    Z_bar = np.mean(Z_jk, axis=0)
    var = (n - 1) / n * np.sum(np.abs(Z_jk - Z_bar) ** 2, axis=0)
    return np.sqrt(var)

def _err_bootstrap(SER_segs, SHR_segs, robust="mean", n_boot=200, rng=None):
    """
    Bootstrap 自举误差估计。

    有放回抽取分段，重复 n_boot 次，
    根据多次 Z 的离散程度估计 sigma_Z。
    """
    n = SER_segs.shape[0]
    if n < 4:
        return np.full((2, 2), np.nan)

    if rng is None:
        rng = np.random.default_rng(0)

    Z_bs = np.empty((n_boot, 2, 2), dtype=complex)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Z_b, _, _ = _Z_from_segments(SER_segs[idx], SHR_segs[idx], robust=robust)
        Z_bs[b] = Z_b

    return np.std(np.abs(Z_bs), axis=0)

def _err_analytic(SER_segs, Z):
    """
    简易解析近似误差。

    注意：这不是严格 Sims-Bostick / Gamble 解析误差，
    这里只是基于分段 SER 散布的近似量。
    正式评价建议优先使用 jackknife 或 bootstrap。
    """
    n = SER_segs.shape[0]
    if n <= 4:
        return np.full((2, 2), np.nan)

    SER_mean = np.mean(SER_segs, axis=0)
    sigma = np.zeros((2, 2), dtype=float)

    for a in range(2):
        for b in range(2):
            spread = np.std(SER_segs[:, a, b]) / max(np.abs(SER_mean[a, b]), 1e-30)
            sigma[a, b] = np.abs(Z[a, b]) * spread / np.sqrt(n)

    return sigma

def estimate_impedance(ts, fs, target_freqs,
                       ts_ref=None,
                       coh_thresh=None,
                       n_cycles=8,
                       min_nperseg=256,
                       overlap=0.5,
                       robust="mean",
                       err="jackknife",
                       n_boot=200):
    """
    对一个频段的 AMT 时间序列估计阻抗。

    参数：
        ts          : 本地时序字典，包含 Ex, Ey, Hx, Hy
        fs          : 采样率
        target_freqs: 目标频率数组
        ts_ref      : 远参考磁场字典，包含 Hx, Hy；None 表示单站法
        coh_thresh  : 频点级多重相干度门限；None 表示不剔除频点
        n_cycles    : 每个窗口包含的周期数
        min_nperseg : 最小窗口长度
        overlap     : 分段重叠比例
        robust      : mean / median / huber
        err         : jackknife / bootstrap / analytic / None
        n_boot      : bootstrap 次数

    返回：
        Z_all   : (nfreq, 2, 2) complex
        sig_all : (nfreq, 2, 2) float，sigma_Z
        info    : dict，包含 f_used, n_segments, n_used, coh_mean, coh_ex, coh_ey
    """
    chans = ("Ex", "Ey", "Hx", "Hy")
    n = min(len(ts[c]) for c in chans)

    data = np.vstack([
        np.asarray(ts[c][:n], dtype=float) - np.mean(ts[c][:n])
        for c in chans
    ])

    # 远参考数据，只需要 Hx、Hy
    use_rr = ts_ref is not None
    if use_rr:
        nr = min(len(ts_ref[c]) for c in ("Hx", "Hy"))
        n = min(n, nr)
        data = data[:, :n]
        ref = np.vstack([
            np.asarray(ts_ref[c][:n], dtype=float) - np.mean(ts_ref[c][:n])
            for c in ("Hx", "Hy")
        ])
    else:
        ref = None

    nf = len(target_freqs)

    Z_all   = np.full((nf, 2, 2), np.nan + 1j * np.nan, dtype=complex)
    sig_all = np.full((nf, 2, 2), np.nan, dtype=float)
    f_used  = np.full(nf, np.nan, dtype=float)

    n_segs = np.zeros(nf, dtype=int)
    n_used = np.zeros(nf, dtype=int)

    coh_mean = np.full(nf, np.nan, dtype=float)
    coh_ex   = np.full(nf, np.nan, dtype=float)
    coh_ey   = np.full(nf, np.nan, dtype=float)

    rng_bs = np.random.default_rng(12345)

    for i, f0 in enumerate(target_freqs):
        if f0 <= 0 or f0 >= fs / 2:
            continue

        # 频率越低，窗口越长；频率越高，窗口越短
        nperseg = max(int(round(fs * n_cycles / f0)), min_nperseg)
        nperseg = min(nperseg, n)

        if nperseg < 32:
            continue

        step = max(int(nperseg * (1 - overlap)), 1)
        starts = np.arange(0, n - nperseg + 1, step)

        if len(starts) < 4:
            continue

        nfft = int(2 ** np.ceil(np.log2(nperseg)))
        freqs_fft = np.fft.rfftfreq(nfft, d=1 / fs)
        k = np.argmin(np.abs(freqs_fft - f0))
        f_used[i] = freqs_fft[k]

        window = get_window("hann", nperseg)
        nseg = len(starts)

        E_segs = np.empty((nseg, 2), dtype=complex)  # [Ex, Ey]
        H_segs = np.empty((nseg, 2), dtype=complex)  # [Hx, Hy] 本地磁场
        R_segs = np.empty((nseg, 2), dtype=complex)  # 参考磁场：单站=H，远参考=Hr

        for j, st in enumerate(starts):
            seg = data[:, st:st + nperseg]
            seg = (seg - seg.mean(axis=1, keepdims=True)) * window[None, :]
            F = np.fft.rfft(seg, n=nfft, axis=1)

            Exf, Eyf, Hxf, Hyf = F[:, k]
            E_segs[j] = [Exf, Eyf]
            H_segs[j] = [Hxf, Hyf]

            if use_rr:
                segr = ref[:, st:st + nperseg]
                segr = (segr - segr.mean(axis=1, keepdims=True)) * window[None, :]
                Fr = np.fft.rfft(segr, n=nfft, axis=1)
                R_segs[j] = Fr[:, k]     # [Hx_ref, Hy_ref]
            else:
                R_segs[j] = [Hxf, Hyf]   # 单站法：R = H

        # ---------- 频点级多重相干度诊断 ----------
        cx, cy = multiple_coherence(E_segs, H_segs)
        coh_ex[i] = cx
        coh_ey[i] = cy
        coh_mean[i] = np.nanmean([cx, cy])

        n_segs[i] = nseg

        # 默认保留所有分段。
        # multiple_coherence 是频点级指标，不用于逐分段筛选。
        keep = np.ones(nseg, dtype=bool)

        # 可选：若设置 coh_thresh，则剔除整体相干度过低的频点。
        # 初期建议 coh_thresh=None，只诊断不剔除。
        if coh_thresh is not None and np.isfinite(coh_mean[i]) and coh_mean[i] < coh_thresh:
            n_used[i] = 0
            continue

        E_k = E_segs[keep]
        H_k = H_segs[keep]
        R_k = R_segs[keep]
        n_used[i] = E_k.shape[0]

        if n_used[i] < 4:
            continue

        # SER = E · R*，SHR = H · R*
        SER_segs = np.einsum("na,nb->nab", E_k, np.conj(R_k))
        SHR_segs = np.einsum("na,nb->nab", H_k, np.conj(R_k))

        Z, _, _ = _Z_from_segments(SER_segs, SHR_segs, robust=robust)

        if err == "jackknife":
            sig = _err_jackknife(SER_segs, SHR_segs, robust=robust)
        elif err == "bootstrap":
            sig = _err_bootstrap(SER_segs, SHR_segs,
                                 robust=robust, n_boot=n_boot, rng=rng_bs)
        elif err == "analytic":
            sig = _err_analytic(SER_segs, Z)
        else:
            sig = np.full((2, 2), np.nan)

        Z_all[i] = Z
        sig_all[i] = sig

    info = {
        "f_used": f_used,
        "n_segments": n_segs,
        "n_used": n_used,
        "coh_mean": coh_mean,
        "coh_ex": coh_ex,
        "coh_ey": coh_ey,
    }

    return Z_all, sig_all, info

def calc_rhoa_phase(freqs, Z, sigma_Z=None, unit="SI"):
    """
    由阻抗张量计算视电阻率和相位。

    unit:
        "SI"    : Z 单位为 (V/m)/(A/m)
                  rho = |Z|^2 / (mu0 * omega)
        "field" : Z 单位为 (mV/km)/nT
                  rho = 0.2 / f * |Z|^2

    误差传播：
        rho = c |Z|^2  -> sigma_rho = 2 c |Z| sigma_Z
        phi = arg(Z)   -> sigma_phi ≈ sigma_Z / |Z|  [rad]
    """
    freqs = np.asarray(freqs)
    Zxy = Z[:, 0, 1]
    Zyx = Z[:, 1, 0]

    if unit == "SI":
        c = 1.0 / (MU0 * 2 * np.pi * freqs)
    elif unit == "field":
        c = 0.2 / freqs
    else:
        raise ValueError("unit must be 'SI' or 'field'")

    rho_xy = c * np.abs(Zxy) ** 2
    rho_yx = c * np.abs(Zyx) ** 2

    phase_xy = (np.degrees(np.angle(Zxy)) + 180) % 180
    phase_yx = (np.degrees(np.angle(-Zyx)) + 180) % 180

    out = {
        "rho_xy": rho_xy,
        "rho_yx": rho_yx,
        "phase_xy": phase_xy,
        "phase_yx": phase_yx,
    }

    if sigma_Z is not None:
        s_xy = sigma_Z[:, 0, 1]
        s_yx = sigma_Z[:, 1, 0]

        out["sigma_rho_xy"] = 2 * c * np.abs(Zxy) * s_xy
        out["sigma_rho_yx"] = 2 * c * np.abs(Zyx) * s_yx
        out["sigma_phase_xy"] = np.degrees(s_xy / np.maximum(np.abs(Zxy), 1e-30))
        out["sigma_phase_yx"] = np.degrees(s_yx / np.maximum(np.abs(Zyx), 1e-30))

    return out

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks

# Parámetros
depth_lim = -200
tamano_ventana_m = 3
min_muestras = 2

# ----------------- utilidades base -----------------
def notch_filter(x, fs=3333, f0=50.0, bw=1.0, harmonics=0, use_fft=True):
    x = np.asarray(x, dtype=float); n = x.size
    if n == 0: return x
    if use_fft and n >= 4:
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        X = np.fft.rfft(x)
        notch_mask = np.zeros_like(freqs, dtype=bool)
        for k in range(harmonics + 1):
            f_c = f0*(k+1)
            if f_c >= fs/2: continue
            half_bw = bw/2.0
            notch_mask |= (freqs >= (f_c-half_bw)) & (freqs <= (f_c+half_bw))
        X[notch_mask] = 0.0
        x = np.fft.irfft(X, n=n)
        b,a=[1.0,0.0],[1.0,0.0]
        padlen = None
        if n <= 3: padlen = n-1
        return signal.filtfilt(b, a, x, padlen=padlen)
    Q = max(1e-3, float(f0)/float(bw)) if bw>0 else 30.0
    w0 = np.clip(f0/(fs/2.0), 1e-6, 1-1e-6)
    b,a = signal.iirnotch(w0, Q)
    return signal.filtfilt(b, a, x)

def aplicar_notch(data, fs, freq, q):
    x = np.asarray(data, dtype=float)
    if x.size == 0: return x
    q = float(q) if q is not None else 30.0
    q = max(q, 1e-6)
    bw = float(freq)/q
    try:
        return notch_filter(x, fs=float(fs), f0=float(freq), bw=float(bw),
                            harmonics=0, use_fft=True)
    except Exception:
        if x.size <= 9: return x
        nyq = 0.5*float(fs)
        w0 = np.clip(float(freq)/nyq, 1e-6, 1-1e-6)
        b,a = signal.iirnotch(w0, q)
        return signal.filtfilt(b, a, x)

def aplicar_un_filtro(data, filter_type, params, fs):
    x = np.asarray(data, dtype=float)
    if x.size == 0: return x
    params = params or {}
    try:
        order  = int(params.get("order", 2))
        cutoff = float(params.get("cutoff", 50.0))
        ripple = float(params.get("ripple", 0.1))
        padlen_min = max(3*(order+1), 9)
        if x.size <= padlen_min: return x
        nyq = 0.5*float(fs)
        wn = np.clip(cutoff/nyq, 1e-6, 0.999999)
        if filter_type.lower() in ("butter",):
            b,a = signal.butter(order, wn, btype='low')
            return signal.filtfilt(b, a, x)
        elif filter_type.lower() in ("cheby",):
            b,a = signal.cheby1(order, ripple, wn, btype='low')
            return signal.filtfilt(b, a, x)
        return x
    except Exception:
        return data

def aplicar_filtros(data, config, fs):
    data = np.asarray(data, float)
    if config["notch"]:
        if len(data) >= 9:
            data = aplicar_notch(data, fs, config["notch"].get("frequency",50),
                                 config["notch"].get("q_value",30))
    return aplicar_un_filtro(data, config["type"], config["params"], fs)

def get_filter_block(json_wrap, key="filters"):
    f = (json_wrap or {}).get(key, {}) or {}
    mode = f.get("mode","global")
    notch = f.get("notch", {"frequency":50, "q_value":30})
    if mode == "global":
        g = f.get("global", {})
        return {"mode":"global","notch":notch,"type":g.get("type","butter"),"params":g.get("params",{})}
    else:
        raise ValueError(f"Modo de filtros '{mode}' no válido")

# ----------------- soporte picos manuales -----------------
def _nearest_index_by_depth(depth_arr, dval):
    depth_arr = np.asarray(depth_arr, float)
    idx = int(np.argmin(np.abs(depth_arr - float(dval))))
    return idx

def preparar_ventanas_manuales(manual_cfg, depth, signal_vals):
    """
    Convierte manual_peaks -> lista de ventanas [{left_idx,right_idx,peak_idx}, ...]
    Soporta:
      - items: {peak}           (depth), busca left/right por cruce automático
      - items: {left,right}     (depth), no requiere peak; peak = máximo en [left,right]
      - items: {peak,left,right}(depth), usa tal cual
    Retorna (ventanas_manuales, peaks_usados) o (None, None) si no hay manuales.
    """
    if not manual_cfg:
        return None, None

    mode = (manual_cfg.get("mode") or "depth").lower()
    if mode != "depth":
        raise ValueError("manual_peaks.mode debe ser 'depth'")

    items = manual_cfg.get("items", [])
    if not items:
        raise ValueError("manual_peaks inválido: items vacío")

    depth_vals = np.asarray(depth, float)
    signal_vals = np.asarray(signal_vals, float)

    ventanas = []
    usados = []

    for it in items:
        has_peak = ("peak" in it)
        has_left = ("left" in it)
        has_right = ("right" in it)

        peak_idx = None
        left_idx = None
        right_idx = None
        peak_depth = None
        target_fraction = None 

        # Caso 1: left/right sin peak → peak = máximo dentro del tramo
        if has_left and has_right and not has_peak:
            left_idx  = _nearest_index_by_depth(depth_vals, it["left"])
            right_idx = _nearest_index_by_depth(depth_vals, it["right"])
            if left_idx > right_idx:
                left_idx, right_idx = right_idx, left_idx
            seg = signal_vals[left_idx:right_idx+1]
            if seg.size == 0:
                continue
            rel = int(np.argmax(seg))
            peak_idx = left_idx + rel
            peak_depth = float(depth_vals[peak_idx])
            target_fraction = None
        # Caso 2: peak solo → buscar cruces de nivel automáticamente
        elif has_peak and not (has_left and has_right):
            peak_idx = _nearest_index_by_depth(depth_vals, it["peak"])
            peak_depth = float(depth_vals[peak_idx])
            peak_val = float(signal_vals[peak_idx])
            # regla de objetivo
            fr = manual_cfg.get("target_rule","auto")
            if isinstance(fr,(int,float)): target_fraction = float(fr)
            else:
                if   peak_val >= 10: target_fraction = 0.8
                elif peak_val >= 5:  target_fraction = 0.75
                elif peak_val > 0:   target_fraction = 0.85
                else:                target_fraction = 0.8
            target = peak_val * target_fraction
            # cruces
            for i in range(peak_idx-1, -1, -1):
                if signal_vals[i] < target:
                    left_idx = i; break
            for i in range(peak_idx+1, len(signal_vals)):
                if signal_vals[i] < target:
                    right_idx = i; break
            if left_idx is None or right_idx is None:
                continue
        # Caso 3: peak + left + right → usar tal cual
        elif has_peak and has_left and has_right:
            left_idx  = _nearest_index_by_depth(depth_vals, it["left"])
            right_idx = _nearest_index_by_depth(depth_vals, it["right"])
            if left_idx > right_idx:
                left_idx, right_idx = right_idx, left_idx
            peak_idx = _nearest_index_by_depth(depth_vals, it["peak"])
            peak_depth = float(depth_vals[peak_idx])
            target_fraction = None
        else:
            continue

        ventanas.append({
            "left_idx":  int(left_idx),
            "right_idx": int(right_idx),
            "peak_idx":  int(peak_idx),
        })
        usados.append({
            "left_idx":  int(left_idx),
            "peak_idx":  int(peak_idx),
            "right_idx": int(right_idx),
            "peak_depth": float(peak_depth) if peak_depth is not None else None,
            "target_fraction": target_fraction
        })

    if not ventanas:
        return [], []  

    return ventanas, usados

# ----------------- caudales -----------------
def procesar_caudales(prediction_filtered, depth, ventanas_manuales=None):
    df_pred = pd.DataFrame({
        'Prediction_filtrada': np.asarray(prediction_filtered, float),
        'Depth (X)': np.asarray(depth, float)
    })
    signal_vals = df_pred['Prediction_filtrada'].values
    depth_vals  = df_pred['Depth (X)'].values

    aquifer_windows = []

    if isinstance(ventanas_manuales, list):
        for w in ventanas_manuales:
            li = int(w["left_idx"]); ri = int(w["right_idx"])
            if li < 0 or ri >= len(depth_vals) or li >= ri:
                continue
            b = abs(depth_vals[ri] - depth_vals[li])
            aquifer_windows.append({
                'Depth Start': depth_vals[li],
                'Depth End':   depth_vals[ri],
                'b (m)':       float(b),
                'Peak Value':  float(signal_vals[int(w["peak_idx"])])
            })
    else:
        # Detección automática tradicional
        all_peaks, _ = find_peaks(signal_vals, prominence=0.55)
        peaks = [idx for idx in all_peaks if depth_vals[idx] < -5]
        for peak_idx in peaks:
            peak_val = signal_vals[peak_idx]
            if   peak_val >= 10: target = peak_val*0.8
            elif peak_val >= 5:  target = peak_val*0.75
            elif peak_val > 0:   target = peak_val*0.85
            else:                continue
            left_idx = None; right_idx = None
            for i in range(peak_idx-1, -1, -1):
                if signal_vals[i] < target:
                    left_idx = i; break
            for i in range(peak_idx+1, len(signal_vals)):
                if signal_vals[i] < target:
                    right_idx = i; break
            if left_idx is None or right_idx is None:
                continue
            b = abs(depth_vals[right_idx] - depth_vals[left_idx])
            aquifer_windows.append({
                'Depth Start': depth_vals[left_idx],
                'Depth End':   depth_vals[right_idx],
                'b (m)':       float(b),
                'Peak Value':  float(peak_val)
            })

    # DataFrame de ventanas
    df_all_windows = pd.DataFrame(aquifer_windows)
    df_all_windows_b = df_pred.copy()
    df_all_windows_b['b_window'] = 0.0

    for _, row in df_all_windows.iterrows():
        start = float(row['Depth Start']); end = float(row['Depth End'])
        lower, upper = min(start,end), max(start,end)
        mask = (df_all_windows_b['Depth (X)'] >= lower) & (df_all_windows_b['Depth (X)'] <= upper)
        df_all_windows_b.loc[mask, 'b_window'] = float(row['b (m)'])

    df_all_windows_b['Q_min'] = df_all_windows_b['Prediction_filtrada'] * 300 * df_all_windows_b['b_window'] * 0.015
    df_all_windows_b['Q_max'] = df_all_windows_b['Prediction_filtrada'] * 900 * df_all_windows_b['b_window'] * 0.015

    conv = 1000/86400.0
    df_all_windows_b['Q_min_Lps'] = df_all_windows_b['Q_min'] * conv
    df_all_windows_b['Q_max_Lps'] = df_all_windows_b['Q_max'] * conv

    prof_min = float(df_all_windows_b['Depth (X)'].min())
    prof_max = float(df_all_windows_b['Depth (X)'].max())
    limites = np.arange(prof_min, prof_max + tamano_ventana_m, tamano_ventana_m)

    resultados_ventanas = []
    for i in range(len(limites)-1):
        z_min = limites[i]; z_max = limites[i+1]
        ventana_df = df_all_windows_b[(df_all_windows_b['Depth (X)'] >= z_min) & (df_all_windows_b['Depth (X)'] < z_max)]
        if len(ventana_df) < min_muestras:
            continue
        resultados_ventanas.append({
            'Profundidad media (m)': float((z_min+z_max)/2.0),
            'Q_min_promedio': float(ventana_df['Q_min_Lps'].mean()),
            'Q_max_promedio': float(ventana_df['Q_max_Lps'].mean()),
        })

    df_resultados_ventanas = pd.DataFrame(resultados_ventanas)
    return {'df_resultados_ventanas': df_resultados_ventanas}

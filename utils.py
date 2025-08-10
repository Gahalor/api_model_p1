import numpy as np
import pandas as pd
import pywt
import requests
from scipy import signal
from scipy.signal import find_peaks

# Variables
depth_lim = -200  # Profundidad maxima para filtrar datos

# Caudal
prominence = 0.8  # Prominencia para detectar peaks
tamano_ventana_m = 3
min_muestras = 2

def safe_get(data, *keys, default=None):
    for k in keys:
        if isinstance(data, dict) and k in data:
            data = data[k]
        else:
            return default
    return data

def read_json_data(json_data):
    """Devuelve (df_signals, metadata) compatible con JSON directo o anidado."""
    se = (
        safe_get(json_data, "seismoelectric", "data") or
        safe_get(json_data, "samples", "seismoelectric", "data") or {}
    )
    geo = safe_get(json_data, "geolocation", default=[])
    sr = (
        safe_get(json_data, "seismoelectric", "sampleRate") or
        safe_get(json_data, "samples", "seismoelectric", "sampleRate") or
        3333
    )
    
    v1 = np.array(se.get("v1", []), float) * 1e-3
    v2 = np.array(se.get("v2", []), float) * 1e-3
    depth = np.array(se.get("deep", []), float)

    n = min(len(v1), len(v2), len(depth))
    df = pd.DataFrame({
        "BLUE": v1[:n],
        "RED":  v2[:n],
        "Depth": depth[:n],
        "lat": [geo[0].get("latitude", 0)] * n if geo else [0] * n,
        "lon": [geo[0].get("longitude", 0)] * n if geo else [0] * n,
    })

    metadata = {"sampling": sr, "geolocation": geo}
    return df, metadata

def notch_filter(x, fs=3333, f0=50.0, bw=1.0, harmonics=0, use_fft=True):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x
    if use_fft and n >= 4:
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        X = np.fft.rfft(x)
        notch_mask = np.zeros_like(freqs, dtype=bool)
        for k in range(harmonics + 1):
            f_c = f0 * (k + 1)
            if f_c >= fs / 2:
                continue
            half_bw = bw / 2.0
            notch_mask |= (freqs >= (f_c - half_bw)) & (freqs <= (f_c + half_bw))
        X[notch_mask] = 0.0
        x = np.fft.irfft(X, n=n)
        # identidad compatible con filtfilt
        b, a = [1.0, 0.0], [1.0, 0.0]
        padlen = None
        if n <= 3:
            padlen = n - 1
        return signal.filtfilt(b, a, x, padlen=padlen)
    # Fallback IIR
    Q = max(1e-3, float(f0) / float(bw)) if bw > 0 else 30.0
    w0 = np.clip(f0 / (fs / 2.0), 1e-6, 1 - 1e-6)
    b, a = signal.iirnotch(w0, Q)
    return signal.filtfilt(b, a, x)

def aplicar_notch(data, fs, freq, q):
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        return x

    # De Q a ancho de banda
    q = float(q) if q is not None else 30.0
    q = max(q, 1e-6)
    bw = float(freq) / q  # bw = f0 / Q

    try:
        # Usa tu notch mejorado (FFT con fallback IIR)
        return notch_filter(x, fs=float(fs), f0=float(freq), bw=float(bw),
                            harmonics=0, use_fft=True)
    except Exception:
        # Último fallback: IIR notch clásico con normalización robusta
        nyq = 0.5 * float(fs)
        w0 = np.clip(float(freq) / nyq, 1e-6, 1 - 1e-6)
        b, a = signal.iirnotch(w0, q)
        # Evita filtfilt en tramos muy cortos
        if x.size <= 9:
            return x
        return signal.filtfilt(b, a, x)



def aplicar_un_filtro(data, filter_type, params, fs):
    """
    Aplica low-pass Butterworth o Chebyshev I.
    - Misma firma/salida.
    - Normaliza corte a [0, 1) respecto a Nyquist.
    - Evita filtfilt en tramos demasiado cortos.
    """
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        return x

    params = params or {}
    try:
        order  = int(params.get("order", 2))
        cutoff = float(params.get("cutoff", 50.0))
        ripple = float(params.get("ripple", 0.1))  # solo cheby

        # Evitar filtfilt cuando no hay muestras suficientes
        padlen_min = max(3 * (order + 1), 9)
        if x.size <= padlen_min:
            return x

        nyq = 0.5 * float(fs)
        wn = np.clip(cutoff / nyq, 1e-6, 0.999999)

        if filter_type.lower() in ("butter"):
            b, a = signal.butter(order, wn, btype='low')
            return signal.filtfilt(b, a, x)

        elif filter_type.lower() in ("cheby"):
            b, a = signal.cheby1(order, ripple, wn, btype='low')
            return signal.filtfilt(b, a, x)

        # Tipo no reconocido → no tocar
        return x

    except Exception:
        # Pase silencioso: conserva entrada ante cualquier error
        return data

def aplicar_filtros(data, config, fs):
    data = np.asarray(data)

    if config["mode"] == "global":
        if config["notch"]:
            if len(data) >= 9:
                data = aplicar_notch(data, fs,
                                     config["notch"].get("frequency", 50),
                                     config["notch"].get("q_value", 30))
            else:
                raise ValueError("Datos muy cortos para aplicar notch. Se omite.")
        return aplicar_un_filtro(data, config["type"], config["params"], fs)

    elif config["mode"] == "by_ranges":
        depth = config.get("depth", None)
        if depth is None:
            raise ValueError("Se requiere un array de depth para modo by_ranges")
        depth = np.asarray(depth)
        if len(depth) != len(data):
            raise ValueError("depth y data deben tener el mismo tamaño")

        filtered = np.zeros_like(data)
        for tramo in config.get("ranges", []):
            mn, mx = tramo["range"]
            mask = (depth >= mn) & (depth < mx)
            tramo_data = data[mask]
            if config["notch"] and len(tramo_data) >= 9:
                tramo_data = aplicar_notch(tramo_data, fs,
                                           config["notch"].get("frequency", 50),
                                           config["notch"].get("q_value", 30))
            filtered[mask] = aplicar_un_filtro(
                tramo_data,
                tramo.get("type", "butter"),
                tramo.get("params", {}),
                fs
            )
        return filtered

    else:
        raise ValueError(f"Modo de filtro '{config['mode']}' no soportado")

def build_features_dataframe(df, meta, v1f, v2f):
    min_len = min(len(v1f), len(v2f), len(df["Depth"]))
    geo = meta.get("geolocation", [])
    latitude = geo[0].get("latitude", 0) if geo else 0
    longitude = geo[0].get("longitude", 0) if geo else 0
    depth = df["Depth"].to_numpy()[:min_len]

    out_df = pd.DataFrame({
        "BLUE Channel (Z)": v1f[:min_len],
        "RED Channel (Z)": v2f[:min_len],
        "latitude": [latitude] * min_len,
        "longitude": [longitude] * min_len,
        "Depth (X)": depth
    })

    out_df = out_df[out_df["Depth (X)"] >= depth_lim]
    out_df['A+B'] = (out_df['BLUE Channel (Z)'] + out_df['RED Channel (Z)']) / 2
    out_df['A-B'] = out_df['BLUE Channel (Z)'] - out_df['RED Channel (Z)']
    out_df['A*B'] = out_df['BLUE Channel (Z)'] * out_df['RED Channel (Z)']
    out_df['A/B'] = np.divide(out_df['BLUE Channel (Z)'], out_df['RED Channel (Z)'],
                              out=np.zeros_like(out_df['BLUE Channel (Z)']),
                              where=out_df['RED Channel (Z)'] != 0)
    out_df['AB'] = np.abs(out_df['BLUE Channel (Z)'] - out_df['RED Channel (Z)'])
    out_df['AAB'] = out_df['BLUE Channel (Z)'] ** 2 - out_df['RED Channel (Z)'] ** 2
    out_df['BAB'] = out_df['BLUE Channel (Z)'] ** 2 + out_df['RED Channel (Z)'] ** 2
    out_df['RAB'] = np.sqrt(out_df['BLUE Channel (Z)'] ** 2 + out_df['RED Channel (Z)'] ** 2)
    out_df['MBA'] = out_df['BLUE Channel (Z)'].rolling(window=2).mean()
    out_df['MBB'] = out_df['RED Channel (Z)'].rolling(window=2).mean()
    out_df['dA'] = out_df['BLUE Channel (Z)'].diff().bfill()
    out_df['dB'] = out_df['RED Channel (Z)'].diff().bfill()
    out_df['ABB'] = out_df['BLUE Channel (Z)'] + out_df['RED Channel (Z)'] ** 2

    return out_df

def get_filter_block(json_data, key="filters"):
    """
    Obtiene la configuración de filtros desde el JSON.
    key: "filters" para voltajes o "filters_magnetometer" para magnetómetro.
    """
    f = json_data.get(key, {})
    mode = f.get("mode", "global")
    notch = f.get("notch", {"frequency": 50, "q_value": 30})

    if mode == "global":
        g = f.get("global", {})
        return {
            "mode": "global",
            "notch": notch,
            "type": g.get("type", "butter"),
            "params": g.get("params", {})
        }
    elif mode == "by_ranges":
        return {
            "mode": "by_ranges",
            "notch": notch,
            "ranges": f.get("ranges", [])
        }
    else:
        raise ValueError(f"Modo de filtros '{mode}' no válido")

def clasificar_caudal(caudal):
    """Clasifica el caudal en diferentes zonas según su valor."""
    if caudal < 0.1:
        return 'Zona no saturada'
    elif 0.1 <= caudal < 0.5:
        return 'Zona de transición'
    elif 0.5 <= caudal < 2.5:
        return 'Zona de caudales muy pequeños'
    elif 2.5 <= caudal < 10:
        return 'Zona de caudal menor'
    elif 10 <= caudal < 25:
        return 'Zona de caudal medio'
    else:
        return 'Zona de caudal mayor'

def procesar_caudales(prediction_filtered, depth):
    """
    Procesa los valores filtrados de predicción para calcular caudales y ventanas.
    
    Args:
        prediction_filtered: array de valores de predicción filtrados
        depth: array de profundidades correspondientes
        
    Returns:
        dict con los dataframes resultantes y estadísticas
    """

    df_pred = pd.DataFrame({
        'Prediction_filtrada': prediction_filtered,
        'Depth (X)': depth
    })
    
    signal = df_pred['Prediction_filtrada'].values
    depth_values = df_pred['Depth (X)'].values
    
    # Recopilar resultados para todos los puntos
    all_aquifer_windows = []
    
    # Detectar picos
    all_peaks, _ = find_peaks(signal, prominence=prominence)
    peaks = [idx for idx in all_peaks if depth_values[idx] < -15]
    
    for peak_idx in peaks:
        peak_val = signal[peak_idx]
        if peak_val >= 10:
            target = peak_val * 0.5
        elif 5 <= peak_val < 10:
            target = peak_val * 0.85
        elif 0 < peak_val < 5:
            target = peak_val * 0.95
        else:
            continue
            
        # Buscar cruce por la izquierda
        left_idx = None
        for i in range(peak_idx - 1, -1, -1):
            if signal[i] < target:
                left_idx = i
                break
                
        # Buscar cruce por la derecha
        right_idx = None
        for i in range(peak_idx + 1, len(signal)):
            if signal[i] < target:
                right_idx = i
                break
                
        if left_idx is None or right_idx is None:
            continue
            
        b = abs(depth_values[right_idx] - depth_values[left_idx])
        all_aquifer_windows.append({
            'Depth Start': depth_values[left_idx],
            'Depth End': depth_values[right_idx],
            'b (m)': b,
            'Peak Value': peak_val
        })
    
    # Crear DataFrame con todos los resultados
    df_all_windows = pd.DataFrame(all_aquifer_windows)
    
    # Asegurar copia del dataframe original
    df_all_windows_b = df_pred.copy()
    
    # Asegurar columna inicializada en ceros
    df_all_windows_b['b_window'] = 0.0
    
    # Recorrer cada ventana y asignar b en las profundidades correspondientes
    for _, row in df_all_windows.iterrows():
        start = row['Depth Start']
        end = row['Depth End']
        b_val = row['b (m)']
        lower = min(start, end)
        upper = max(start, end)
        mask = (
            (df_all_windows_b['Depth (X)'] >= lower) &
            (df_all_windows_b['Depth (X)'] <= upper)
        )
        df_all_windows_b.loc[mask, 'b_window'] = b_val
    
    # Crear columnas de caudal
    df_all_windows_b['Q_min'] = df_all_windows_b['Prediction_filtrada'] * 10 * df_all_windows_b['b_window'] * 0.015
    df_all_windows_b['Q_mean'] = df_all_windows_b['Prediction_filtrada'] * 500.01 * df_all_windows_b['b_window'] * 0.015
    df_all_windows_b['Q_max'] = df_all_windows_b['Prediction_filtrada'] * 1000 * df_all_windows_b['b_window'] * 0.015
    
    # Conversión de m³/día a L/s
    conversion_factor = 1000 / 86400  # ≈ 0.0115741
    
    # Crear columnas con caudal en litros/segundo
    df_all_windows_b['Q_min_Lps'] = df_all_windows_b['Q_min'] * conversion_factor
    df_all_windows_b['Q_mean_Lps'] = df_all_windows_b['Q_mean'] * conversion_factor
    df_all_windows_b['Q_max_Lps'] = df_all_windows_b['Q_max'] * conversion_factor
    
    # Aplicar clasificación de caudales
    df_caudal_clasificado = df_all_windows_b[['Depth (X)', 'Q_min_Lps', 'Q_mean_Lps', 'Q_max_Lps']].copy()
    df_caudal_clasificado['Q_min_Lps_clas'] = df_caudal_clasificado['Q_min_Lps'].apply(clasificar_caudal)
    df_caudal_clasificado['Q_mean_Lps_clas'] = df_caudal_clasificado['Q_mean_Lps'].apply(clasificar_caudal)
    df_caudal_clasificado['Q_max_Lps_clas'] = df_caudal_clasificado['Q_max_Lps'].apply(clasificar_caudal)
    
    # Procesamiento por ventanas
    prof_min = df_caudal_clasificado['Depth (X)'].min()
    prof_max = df_caudal_clasificado['Depth (X)'].max()
    
    limites_ventanas = np.arange(prof_min, prof_max + tamano_ventana_m, tamano_ventana_m)
    resultados_ventanas = []
    
    for i in range(len(limites_ventanas) - 1):
        z_min = limites_ventanas[i]
        z_max = limites_ventanas[i + 1]
        
        ventana_df = df_all_windows_b[(df_all_windows_b['Depth (X)'] >= z_min) & (df_all_windows_b['Depth (X)'] < z_max)]
        if len(ventana_df) < min_muestras:
            continue
            
        resumen = {
            'Profundidad media (m)': (z_min + z_max) / 2,
            'Q_max_promedio': ventana_df['Q_max_Lps'].mean(),
        }
        resultados_ventanas.append(resumen)
    
    df_resultados_ventanas = pd.DataFrame(resultados_ventanas)
    
    return {
        'df_resultados_ventanas': df_resultados_ventanas
    }
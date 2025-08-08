import numpy as np
import pandas as pd
import pywt
import requests
from scipy import signal

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

    v1 = np.array(se.get("v1", []), float)
    v2 = np.array(se.get("v2", []), float)
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

def aplicar_notch(data, fs, freq, q):
    b, a = signal.iirnotch(freq, Q=q, fs=fs)
    return signal.filtfilt(b, a, data)

def aplicar_un_filtro(data, filter_type, params, fs):
    """Aplica un filtro seguro, evitando errores con tramos cortos."""
    if params is None:
        params = {}

    try:
        padlen_min = 3 * params.get("order", 5)
        if len(data) <= padlen_min:
            return data

        if filter_type == "butter":
            cutoff = params.get("cutoff", 100)
            order = params.get("order", 5)
            b, a = signal.butter(order, cutoff / (fs / 2), btype='low')
            return signal.filtfilt(b, a, data)

        elif filter_type == "cheby":
            cutoff = params.get("cutoff", 50)
            order = params.get("order", 2)
            ripple = params.get("ripple", 0.5)
            b, a = signal.cheby1(order, ripple, cutoff / (fs / 2), btype='low')
            return signal.filtfilt(b, a, data)

        elif filter_type.lower() == "savgol":
            w = params.get("window_length", 15)
            p = params.get("polyorder", 3)
            if w % 2 == 0:
                w += 1
            return signal.savgol_filter(data, w, p)

        elif filter_type.lower() == "fft":
            ratio = params.get("cutoff_ratio", 0.01)
            F = np.fft.fft(data)
            N = len(F)
            cut = int(N * ratio)
            F[cut:-cut] = 0
            return np.real(np.fft.ifft(F))

        elif filter_type.lower() == "wavelet":
            wl = params.get("wavelet", "db4")
            lvl = params.get("level", 5)
            coeffs = pywt.wavedec(data, wl, level=lvl)
            for i in range(1, len(coeffs)):
                coeffs[i] = np.zeros_like(coeffs[i])
            return pywt.waverec(coeffs, wl)[:len(data)]

        else:
            return data

    except Exception as e:
        pass
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

    out_df = out_df[out_df["Depth (X)"] >= -200]
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

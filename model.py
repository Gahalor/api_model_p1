from flask import Flask, request, jsonify
from functools import wraps
import numpy as np
import os
from utils import (
    get_filter_block, aplicar_filtros, procesar_caudales,
    preparar_ventanas_manuales
)

app = Flask(__name__)


def require_internal_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        expected = os.environ.get("INTERNAL_API_KEY", "")
        if not expected:
            return jsonify({"status": "error", "message": "API no configurada."}), 500
        provided = request.headers.get("X-Internal-Key", "")
        if not provided or provided != expected:
            return jsonify({"status": "error", "message": "No autorizado."}), 401
        return f(*args, **kwargs)
    return decorated


@app.route("/", methods=["POST"])
@require_internal_key
def filter_prediction():
    try:
        data = request.get_json(force=True)
        prediction = np.asarray(data.get("prediction", []), float)
        depth = np.asarray(data.get("depth", []), float)
        filters_config = data.get("filters", {})
        fs = float(data.get("sampling", 3333))

        if prediction.size == 0:
            return jsonify(status="error", error="No se recibieron valores de prediction"), 400
        if depth.size == 0:
            return jsonify(status="error", error="No se recibieron valores de depth"), 400
        if prediction.size != depth.size:
            return jsonify(status="error", error="Los arrays de prediction y depth deben tener el mismo tamaño"), 400

        # Filtros
        config = get_filter_block({"filters": filters_config})
        filtered = aplicar_filtros(prediction, config, fs)
        min_pred = float(np.min(filtered))
        filtered = filtered - min_pred + 0.02

        manual_cfg = (filters_config or {}).get("manual_peaks", None)
        ventanas_manuales, peaks_usados = preparar_ventanas_manuales(
            manual_cfg, depth, filtered
        )

        resultados = procesar_caudales(
            prediction_filtered=filtered,
            depth=depth,
            ventanas_manuales=ventanas_manuales 
        )

        response = {
            "status": "success",
            "prediction_original": prediction.tolist(),
            "prediction_filtered": filtered.tolist(),
            "depth": depth.tolist(),
            "caudales": resultados['df_resultados_ventanas'].to_dict('records'),
        }
        return jsonify(response)

    except Exception as e:
        return jsonify(status="error", error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5400, debug=True)

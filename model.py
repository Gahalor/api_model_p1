
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import os
from utils import get_filter_block, aplicar_filtros

app = Flask(__name__)


@app.route("/", methods=["POST"])
def filter_prediction():
    """
    API para filtrar los valores de prediction usando los filtros definidos en utils.py.
    Espera un JSON con:
      - prediction: array de valores
      - filters: configuración de filtros
      - sampling: frecuencia de muestreo 
      - depth: array de profundidad (solo si se usa modo by_ranges)
    """
    try:
        data = request.get_json(force=True)
        prediction = np.asarray(data.get("prediction", []))
        depth = np.asarray(data.get("depth", []))
        filters_config = data.get("filters", {})
        fs = data.get("sampling", 3333)
        
        if len(prediction) == 0:
            return jsonify(status="error", error="No se recibieron valores de prediction"), 400
        
        if len(depth) == 0:
            return jsonify(status="error", error="No se recibieron valores de depth"), 400
            
        if len(prediction) != len(depth):
            return jsonify(status="error", error="Los arrays de prediction y depth deben tener el mismo tamaño"), 400
        
        config = get_filter_block({"filters": filters_config})
        
        if config["mode"] == "by_ranges":
            config["depth"] = depth

        filtered = aplicar_filtros(prediction, config, fs)
        
        response = {
            "status": "success",
            "prediction_original": prediction.tolist(),
            "prediction_filtered": filtered.tolist(),
            "depth": depth.tolist(),
        }
        return jsonify(response)

    except Exception as e:
        return jsonify(status="error", error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5310, debug=True)

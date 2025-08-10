from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging

from services.model_service import load_models, sample_from_latent, optimize_with_rl, evaluate_batch
from services.props import calculate_properties

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chemgen-backend")

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

logger.info("Loading models...")
models = load_models()
logger.info("Models loaded.")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": models.get("vae") is not None})

@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    z = data.get("z", None)
    n = int(data.get("n", 1))
    try:
        res = sample_from_latent(models, z=z, n=n)
        return jsonify(res)
    except Exception as e:
        logger.exception("Generate error")
        return jsonify({"error": str(e)}), 500

@app.route("/api/props", methods=["POST"])
def props():
    data = request.get_json(force=True)
    smiles_list = data.get("smiles", [])
    try:
        out = [calculate_properties(smi) for smi in smiles_list]
        return jsonify(out)
    except Exception as e:
        logger.exception("Props error")
        return jsonify({"error": str(e)}), 500

@app.route("/api/rl_optimize", methods=["POST"])
def rl_optimize():
    data = request.get_json(force=True)
    target = data.get("target", {"qed": 0.8, "sa_max": 6.0})
    steps = int(data.get("steps", 20000))
    try:
        res = optimize_with_rl(models, target=target, total_timesteps=steps)
        return jsonify({"status": "completed", "result": res})
    except Exception as e:
        logger.exception("RL optimize error")
        return jsonify({"error": str(e)}), 500

@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json(force=True)
    n = int(data.get("n", 32))
    try:
        res = evaluate_batch(models, n=n)
        return jsonify(res)
    except Exception as e:
        logger.exception("Evaluate error")
        return jsonify({"error": str(e)}), 500

# Serve frontend build (if present)
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    index = os.path.join(app.static_folder, "index.html")
    if os.path.exists(index):
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({"message": "Frontend not built. Use /api endpoints."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

"""
MorphAI Flask backend.

Exposes the LGNet facial-inpainting pipeline from ../MorphAI over HTTP and
(optionally) serves the built React dashboard from ../frontend/dist.

    python backend/app.py            # http://127.0.0.1:5000

Routes
------
GET  /api/health      pipeline + device status
GET  /api/variants    available mask variants
GET  /api/evaluation  dataset-level metrics from samples/final_eval_results.json
POST /api/landmarks   MediaPipe landmark overlay for an uploaded face
POST /api/predict     full inpainting pass -> 6 image panels + metrics
POST /predict         legacy shape used by MorphAI/frontend/script.js
"""

from __future__ import annotations

import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import config
from morphai_service import ALL_VARIANTS, VARIANT_LABELS, get_service


def _bool_field(name: str, default: bool = False) -> bool:
    raw = request.form.get(name)
    if raw is None:
        raw = (request.json or {}).get(name) if request.is_json else None
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _read_upload(field_names):
    """Returns (bytes, filename) for the first populated file field."""
    for name in field_names:
        file = request.files.get(name)
        if file and file.filename:
            suffix = Path(file.filename).suffix.lower()
            if suffix and suffix not in config.ALLOWED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
                )
            data = file.read()
            if not data:
                raise ValueError("The uploaded file is empty.")
            return data, file.filename
    return None, None


def create_app() -> Flask:
    # static_folder is disabled so Flask's built-in /<path:filename> rule does not
    # shadow the SPA catch-all below; the bundle is served explicitly instead.
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/predict": {"origins": "*"}})

    # ---------------- API ----------------
    @app.get("/api/health")
    def health():
        return jsonify(get_service().status())

    @app.get("/api/variants")
    def variants():
        return jsonify(
            {
                "variants": [
                    {
                        "id": v,
                        "label": VARIANT_LABELS.get(v, v),
                        "group": "Semantic region" if v.startswith("semantic_") else "Irregular blob",
                    }
                    for v in ALL_VARIANTS
                ],
                "default": "semantic_nose",
                "blends": [
                    {"id": "feather", "label": "Fast alpha feathering"},
                    {"id": "seamless", "label": "Seamless Poisson clone"},
                ],
            }
        )

    @app.get("/api/evaluation")
    def evaluation():
        return jsonify(get_service().evaluation_report())

    @app.post("/api/landmarks")
    def landmarks():
        try:
            image_bytes, _ = _read_upload(["image", "file"])
            if image_bytes is None:
                return jsonify({"success": False, "error": "No image uploaded (field 'image')."}), 400
            return jsonify(get_service().detect_landmarks(image_bytes))
        except ValueError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": f"Landmark detection failed: {exc}"}), 500

    @app.post("/api/predict")
    def predict():
        try:
            image_bytes, filename = _read_upload(["image", "file"])
            if image_bytes is None:
                return jsonify({"success": False, "error": "No image uploaded (field 'image')."}), 400

            mask_bytes, _ = _read_upload(["mask"])

            variant = (request.form.get("variant") or "semantic_nose").strip()
            if not mask_bytes and variant not in ALL_VARIANTS:
                return jsonify(
                    {"success": False, "error": f"Unknown variant '{variant}'. Valid: {', '.join(ALL_VARIANTS)}"}
                ), 400

            result = get_service().predict(
                image_bytes,
                variant=variant,
                fast_blend=(request.form.get("blend", "feather") != "seamless"),
                compute_identity=_bool_field("compute_identity", True),
                custom_mask_bytes=mask_bytes,
            )
            result["filename"] = filename
            return jsonify(result)
        except ValueError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"success": False, "error": str(exc)}), 503
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": f"Inference failed: {exc}"}), 500

    # ---------------- Legacy route ----------------
    # Keeps the original static page (MorphAI/frontend/script.js) working.
    @app.post("/predict")
    def predict_legacy():
        try:
            image_bytes, _ = _read_upload(["image", "file"])
            if image_bytes is None:
                return jsonify({"success": False, "error": "No image uploaded."}), 400

            result = get_service().predict(
                image_bytes,
                variant=(request.form.get("variant") or "semantic_nose").strip(),
                fast_blend=(request.form.get("blend", "feather") != "seamless"),
            )
            identity = result["metrics"]["identity"]
            return jsonify(
                {
                    "success": True,
                    "generated_image": result["images"]["reconstructed"].split(",", 1)[1],
                    "masked_image": result["images"]["masked"].split(",", 1)[1],
                    "confidence": "--" if identity is None else round(identity * 100, 2),
                    "processing_time": result["processing_time"],
                    "ssim": result["metrics"]["ssim"],
                    "psnr": result["metrics"]["psnr"],
                    # FID is a distribution-level metric; it is not defined for a
                    # single image and was not part of the recorded evaluation run.
                    "fid": "n/a",
                }
            )
        except ValueError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"success": False, "error": str(exc)}), 503
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"success": False, "error": str(exc)}), 500

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"success": False, "error": f"Image exceeds {config.MAX_UPLOAD_MB} MB."}), 413

    # ---------------- Static React bundle ----------------
    @app.get("/")
    @app.get("/<path:requested>")
    def serve_spa(requested: str = ""):
        # Never let an unknown /api path fall through to the SPA shell.
        if requested.startswith("api/"):
            return jsonify({"success": False, "error": f"Unknown API route '/{requested}'."}), 404

        index = config.FRONTEND_DIST / "index.html"
        if not index.is_file():
            return jsonify(
                {
                    "service": "MorphAI backend",
                    "message": "React bundle not built. Run `npm run build` in frontend/, "
                    "or use the Vite dev server on http://localhost:5173.",
                    "api": ["/api/health", "/api/variants", "/api/evaluation", "/api/landmarks", "/api/predict"],
                }
            )

        root = str(config.FRONTEND_DIST)
        if requested:
            try:
                candidate = (config.FRONTEND_DIST / requested).resolve()
                # Reject anything that escapes the bundle directory.
                candidate.relative_to(config.FRONTEND_DIST.resolve())
                if candidate.is_file():
                    return send_from_directory(root, requested)
            except (ValueError, OSError):
                pass
        return send_from_directory(root, "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    # Warm the pipeline before the server starts accepting requests so the first
    # prediction is not delayed by the checkpoint load.
    service = get_service()
    print(f"[MorphAI] Model loaded: {service.ready} | device: {service.device}")
    print(f"[MorphAI] Listening on http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, use_reloader=False, threaded=True)

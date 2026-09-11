from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2
from flask_cors import CORS
import time
import pandas as pd
import base64
from PIL import Image
import io
import numpy as np


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    static_url_path="/static",
    static_folder="static"
)

CORS(app)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# NEW 11-CLASS WEAPON DETECTION MODEL
# best.pt must be in the same folder as app.py
# ============================================================

MODEL_PATH = os.path.join(REPO_ROOT, "best.pt")

# Old model - only used as fallback
OLD_MODEL_PATH = os.path.join(REPO_ROOT, "yolov8s.pt")


# ============================================================
# LOAD YOLO MODEL
# ============================================================

try:

    if os.path.exists(MODEL_PATH):

        model = YOLO(MODEL_PATH)

        print("=" * 60)
        print("NEW 11-CLASS WEAPON DETECTION MODEL LOADED")
        print("=" * 60)
        print(f"Model path: {MODEL_PATH}")
        print(f"Classes: {model.names}")
        print("=" * 60)

    elif os.path.exists(OLD_MODEL_PATH):

        print("WARNING: best.pt NOT FOUND")
        print("Loading old yolov8s.pt as fallback")

        model = YOLO(OLD_MODEL_PATH)

        print(f"Model path: {OLD_MODEL_PATH}")
        print(f"Classes: {model.names}")

    else:

        print("ERROR: No YOLO model found")
        model = None

except Exception as e:

    print("=" * 60)
    print("MODEL LOADING ERROR")
    print("=" * 60)
    print(str(e))
    print("=" * 60)

    model = None


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = os.path.join(
    REPO_ROOT,
    "uploads"
)

OUTPUT_FOLDER = os.path.join(
    REPO_ROOT,
    "outputs"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# STATIC / HTML ROUTES
# ============================================================

@app.route("/style.css")
def serve_style_css():

    return send_from_directory(
        app.static_folder,
        "style.css"
    )


@app.route("/", methods=["GET"])
def index():

    return render_template(
        "index.html"
    )


@app.route("/about.html", methods=["GET"])
def about():

    return render_template(
        "about.html"
    )


@app.route("/contact")
def contact():

    return render_template(
        "contact.html"
    )


@app.route("/results.html")
def results_partial():

    return render_template(
        "results.html"
    )


@app.route("/outputs/<filename>")
def serve_output(filename):

    return send_from_directory(
        OUTPUT_FOLDER,
        filename
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "server": "running",

        "model_loaded": model is not None,

        "model_path": (
            MODEL_PATH
            if model is not None
            else None
        ),

        "classes": (
            model.names
            if model is not None
            else None
        )

    }), 200


# ============================================================
# IMAGE DETECTION
# ============================================================

@app.route("/detect", methods=["POST"])
def detect():

    try:

        print("\n" + "=" * 60)
        print("NEW DETECTION REQUEST")
        print("=" * 60)


        # ----------------------------------------------------
        # Check uploaded file
        # ----------------------------------------------------

        if "image" not in request.files:

            print("ERROR: No image field")

            return jsonify({

                "error":
                "No file part. Please send multipart/form-data with field 'image'."

            }), 400


        file = request.files["image"]


        if file.filename == "":

            print("ERROR: Empty filename")

            return jsonify({

                "error":
                "Empty filename."

            }), 400


        filename = secure_filename(
            file.filename
        )


        save_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )


        print(
            f"Uploaded file: {filename}"
        )


        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        file.save(
            save_path
        )


        print(
            f"Saved image: {save_path}"
        )


        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if model is None:

            print(
                "ERROR: Model is not loaded"
            )

            return jsonify({

                "detections": [],

                "filename": filename,

                "annotated_image": None,

                "error":
                "Object detection model not loaded on the server."

            }), 500


        # ----------------------------------------------------
        # Read confidence
        # ----------------------------------------------------

        try:

            confidence_threshold = float(
                request.form.get(
                    "confidence",
                    0.20
                )
            )

        except ValueError:

            confidence_threshold = 0.20


        print(
            f"Confidence threshold: {confidence_threshold}"
        )


        # ----------------------------------------------------
        # Read image BEFORE YOLO
        # ----------------------------------------------------

        img = cv2.imread(
            save_path
        )


        if img is None:

            print(
                "ERROR: OpenCV could not read image"
            )

            return jsonify({

                "error":
                f"Could not read uploaded image: {filename}"

            }), 400


        print(
            f"Image size: {img.shape}"
        )


        # ----------------------------------------------------
        # YOLO prediction
        # ----------------------------------------------------

        print(
            "Running YOLO prediction..."
        )


        results = model.predict(

            source=save_path,

            imgsz=640,

            conf=confidence_threshold,

            verbose=False

        )


        print(
            "YOLO prediction completed"
        )


        # ----------------------------------------------------
        # Process detections
        # ----------------------------------------------------

        detections = []


        for r in results:

            if r.boxes is None:

                continue


            for box in r.boxes:

                cls_id = int(
                    box.cls[0]
                )


                cls_name = model.names[
                    cls_id
                ]


                conf = float(
                    box.conf[0]
                )


                bbox = box.xyxy[
                    0
                ].tolist()


                detections.append({

                    "class":
                    cls_name,

                    "class_id":
                    cls_id,

                    "confidence":
                    conf,

                    "bbox":
                    bbox

                })


                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    bbox
                )


                color = (
                    0,
                    255,
                    0
                )


                cv2.rectangle(

                    img,

                    (x1, y1),

                    (x2, y2),

                    color,

                    2

                )


                # ------------------------------------------------
                # Label
                # ------------------------------------------------

                label = (
                    f"{cls_name} "
                    f"{conf:.2f}"
                )


                cv2.putText(

                    img,

                    label,

                    (
                        x1,
                        max(
                            y1 - 10,
                            20
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    color,

                    2

                )


                print(
                    f"Detected: {cls_name} "
                    f"confidence={conf:.2f}"
                )


        # ----------------------------------------------------
        # Save annotated image
        # ----------------------------------------------------

        output_filename = (
            "annotated_" + filename
        )


        output_path = os.path.join(

            OUTPUT_FOLDER,

            output_filename

        )


        success = cv2.imwrite(

            output_path,

            img

        )


        if not success:

            print(
                "WARNING: Could not save annotated image"
            )


        print(
            f"Annotated image: {output_path}"
        )


        annotated_image_url = (
            f"/outputs/{output_filename}"
        )


        # ----------------------------------------------------
        # Create CSV
        # ----------------------------------------------------

        csv_url = None


        if detections:

            df = pd.DataFrame(
                detections
            )


            csv_filename = (

                "predictions_"

                + filename.rsplit(
                    ".",
                    1
                )[0]

                + ".csv"

            )


            csv_path = os.path.join(

                OUTPUT_FOLDER,

                csv_filename

            )


            df.to_csv(

                csv_path,

                index=False

            )


            csv_url = (
                f"/outputs/{csv_filename}"
            )


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        print(
            f"Total detections: {len(detections)}"
        )


        print(
            "Detection request completed"
        )


        return jsonify({

            "detections":
            detections,

            "count":
            len(detections),

            "filename":
            filename,

            "annotated_image":
            annotated_image_url,

            "csv_url":
            csv_url

        }), 200


    except Exception as e:

        print("\n" + "=" * 60)

        print(
            "DETECTION ERROR"
        )

        print(
            repr(e)
        )

        print("=" * 60)


        return jsonify({

            "error":
            str(e)

        }), 500


# ============================================================
# LIVE WEBCAM DETECTION
# ============================================================

def generate_frames():

    camera_index = int(

        os.environ.get(
            "CAMERA_INDEX",
            0
        )

    )


    camera = cv2.VideoCapture(
        camera_index
    )


    if not camera.isOpened():

        raise RuntimeError(

            f"Could not open webcam index "
            f"{camera_index}."

        )


    print(
        f"Webcam started: index {camera_index}"
    )


    try:

        while True:

            success, frame = camera.read()


            if not success:

                break


            if model is not None:

                results = model.predict(

                    source=frame,

                    imgsz=640,

                    conf=0.30,

                    verbose=False

                )


                for r in results:

                    if r.boxes is None:

                        continue


                    for box in r.boxes:

                        cls_id = int(
                            box.cls[0]
                        )


                        cls_name = model.names[
                            cls_id
                        ]


                        conf = float(
                            box.conf[0]
                        )


                        bbox = box.xyxy[
                            0
                        ].tolist()


                        x1, y1, x2, y2 = map(
                            int,
                            bbox
                        )


                        color = (
                            0,
                            255,
                            0
                        )


                        cv2.rectangle(

                            frame,

                            (x1, y1),

                            (x2, y2),

                            color,

                            2

                        )


                        label = (

                            f"{cls_name} "
                            f"{conf:.2f}"

                        )


                        cv2.putText(

                            frame,

                            label,

                            (
                                x1,
                                max(
                                    y1 - 10,
                                    20
                                )
                            ),

                            cv2.FONT_HERSHEY_SIMPLEX,

                            0.7,

                            color,

                            2

                        )


            ret, buffer = cv2.imencode(

                ".jpg",

                frame

            )


            if not ret:

                continue


            frame_bytes = (
                buffer.tobytes()
            )


            yield (

                b"--frame\r\n"

                b"Content-Type: image/jpeg\r\n\r\n"

                + frame_bytes

                + b"\r\n"

            )


            time.sleep(
                0.03
            )


    finally:

        camera.release()

        print(
            "Webcam released"
        )


@app.route("/live")
def live():

    if model is None:

        return jsonify({

            "error":
            "Model not loaded."

        }), 500


    return Response(

        generate_frames(),

        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )

    )


# ============================================================
# LIVE FRAME DETECTION
# ============================================================

@app.route(
    "/detect_live_frame",
    methods=["POST"]
)
def detect_live_frame():

    try:

        data = request.get_json()


        if not data or "frame" not in data:

            return jsonify({

                "error":
                "No frame data"

            }), 400


        if model is None:

            return jsonify({

                "error":
                "Model not loaded"

            }), 500


        frame_data = data[
            "frame"
        ]


        confidence = float(

            data.get(
                "confidence",
                0.40
            )

        )


        # ----------------------------------------------------
        # Base64 → Image
        # ----------------------------------------------------

        if "," in frame_data:

            frame_data = (
                frame_data.split(
                    ",",
                    1
                )[1]
            )


        image_data = base64.b64decode(
            frame_data
        )


        image_pil = Image.open(

            io.BytesIO(
                image_data
            )

        )


        image_np = np.array(
            image_pil
        )


        frame = cv2.cvtColor(

            image_np,

            cv2.COLOR_RGB2BGR

        )


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model.predict(

            source=frame,

            imgsz=640,

            conf=confidence,

            verbose=False

        )


        detections = []


        for r in results:

            if r.boxes is None:

                continue


            for box in r.boxes:

                cls_id = int(
                    box.cls[0]
                )


                cls_name = model.names[
                    cls_id
                ]


                conf = float(
                    box.conf[0]
                )


                bbox = box.xyxy[
                    0
                ].tolist()


                if conf >= confidence:

                    detections.append({

                        "class":
                        cls_name,

                        "class_id":
                        cls_id,

                        "confidence":
                        conf,

                        "bbox": [

                            int(x)

                            for x in bbox

                        ]

                    })


        return jsonify({

            "detections":
            detections,

            "count":
            len(detections),

            "frame_processed":
            True

        })


    except Exception as e:

        print(
            "Live detection error:",
            repr(e)
        )


        return jsonify({

            "error":
            str(e)

        }), 500


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("STARTING WEAPON DETECTION SERVER")
    print("=" * 60)

    print(
        f"Using model: {MODEL_PATH}"
    )


    if model is not None:

        print(
            f"Classes: {model.names}"
        )

    else:

        print(
            "WARNING: Model is NOT loaded!"
        )


    print("=" * 60)
    print(
        "Server: http://127.0.0.1:8000"
    )
    print("=" * 60)


    app.run(

        host="127.0.0.1",

        port=8000,

        debug=False,

        use_reloader=False

    )
import os
import socket
import base64
from io import BytesIO

import eventlet
eventlet.monkey_patch()

import numpy as np
import cv2
from PIL import Image

from flask import Flask, render_template
import socketio

from keras.models import load_model


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.path.join("model", "model.h5")

START_PORT = 4567
PORT_SEARCH_LIMIT = 20

SPEED_LIMIT = 20.0

# Maximum steering angle displayed on dashboard
MAX_STEERING_ANGLE = 25.0


# ============================================================
# SOCKET.IO
# ============================================================

sio = socketio.Server(
    async_mode="eventlet",
    cors_allowed_origins="*"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    static_folder="resources",
    template_folder="templates"
)


# ============================================================
# MODEL
# ============================================================

model = None


# ============================================================
# HELPERS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def get_available_port(start_port=4567, limit=20):

    for port in range(start_port, start_port + limit):

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:

            try:
                sock.bind(("", port))
                return port

            except OSError:
                continue

    raise OSError(
        f"No free port found in range "
        f"{start_port}-{start_port + limit - 1}"
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def img_preprocess(img):

    # Crop
    img = img[60:135, :, :]

    # RGB -> YUV
    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2YUV
    )

    # Blur
    img = cv2.GaussianBlur(
        img,
        (3, 3),
        0
    )

    # Resize
    img = cv2.resize(
        img,
        (200, 66)
    )

    # Normalize
    img = img.astype(np.float32) / 255.0

    return img


# ============================================================
# PREDICT STEERING
# ============================================================

def predict_steering(image):

    global model

    image = img_preprocess(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    steering_angle = float(
        prediction[0][0]
    )

    return steering_angle


# ============================================================
# THROTTLE
# ============================================================

def calculate_throttle(speed, steering_angle):

    throttle = 1.0 - (
        speed / SPEED_LIMIT
    )

    # Slow down while turning
    if abs(steering_angle) > 0.4:

        throttle *= 0.5

    elif abs(steering_angle) > 0.25:

        throttle *= 0.7

    elif abs(steering_angle) > 0.15:

        throttle *= 0.85

    return clamp(
        throttle,
        0.0,
        1.0
    )


# ============================================================
# SEND CONTROL
# ============================================================

def send_control(
    steering_angle,
    throttle
):

    # Send command to simulator
    sio.emit(
        "steer",
        data={
            "steering_angle": str(
                steering_angle
            ),
            "throttle": str(
                throttle
            )
        }
    )

    # Send steering information to dashboard
    sio.emit(
        "dashboard",
        data={
            "steering_angle": steering_angle,
            "throttle": throttle
        }
    )


# ============================================================
# TELEMETRY
# ============================================================

@sio.on("telemetry")
def telemetry(sid, data):

    try:

        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        speed = max(
            0.0,
            float(data["speed"])
        )


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        if "image" not in data:

            print("No image received")

            send_control(
                0.0,
                0.0
            )

            return


        # Base64 -> bytes
        image_data = base64.b64decode(
            data["image"]
        )


        # bytes -> PIL
        image = Image.open(
            BytesIO(image_data)
        )


        # PIL -> NumPy
        image = np.asarray(
            image
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        steering_angle = predict_steering(
            image
        )


        # ----------------------------------------------------
        # THROTTLE
        # ----------------------------------------------------

        throttle = calculate_throttle(
            speed,
            steering_angle
        )


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"Steering Angle: "
            f"{steering_angle:.3f} | "
            f"Throttle: "
            f"{throttle:.3f} | "
            f"Speed: "
            f"{speed:.2f}"
        )


        # ----------------------------------------------------
        # SEND
        # ----------------------------------------------------

        send_control(
            steering_angle,
            throttle
        )


    except Exception as e:

        print(
            "Telemetry error:",
            e
        )

        send_control(
            0.0,
            0.0
        )


# ============================================================
# CONNECT
# ============================================================

@sio.on("connect")
def connect(sid, environ):

    print(
        "Simulator connected."
    )

    send_control(
        0.0,
        0.0
    )


# ============================================================
# DISCONNECT
# ============================================================

@sio.on("disconnect")
def disconnect(sid):

    print(
        "Simulator disconnected."
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    return render_template(
        "index.html"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SELF DRIVING CAR")
    print("=" * 60)


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "Loading model..."
    )

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    print(
        "Model loaded successfully."
    )

    print(
        "Input shape:",
        model.input_shape
    )

    print(
        "Output shape:",
        model.output_shape
    )


    # --------------------------------------------------------
    # PORT
    # --------------------------------------------------------

    port = get_available_port(
        START_PORT,
        PORT_SEARCH_LIMIT
    )


    # --------------------------------------------------------
    # SOCKET.IO APP
    # --------------------------------------------------------

    socket_app = socketio.WSGIApp(
        sio,
        app.wsgi_app
    )


    # --------------------------------------------------------
    # SERVER
    # --------------------------------------------------------

    print(
        f"\nDashboard:"
        f"\nhttp://localhost:{port}"
    )

    print(
        "\nWaiting for simulator..."
    )


    eventlet.wsgi.server(
        eventlet.listen(
            ("", port)
        ),
        socket_app
    )
    
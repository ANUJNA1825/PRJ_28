"""
Microplastic Pollution Detector - Flask Web App
=================================================
Routes:
    /          -> redirects to login (or predict page if already signed in)
    /signup    -> create an account
    /login     -> sign in
    /logout    -> clear session
    /predict   -> upload an image, get the predicted pollution level

Run:
    py -m pip install flask tensorflow pillow werkzeug
    py app.py

Users are stored in a local JSON file (users.json) for simplicity —
swap this out for a real database before deploying anywhere public.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret-key"  # required for sessions

USERS_FILE = "users.json"
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}
MODEL_PATH = "best_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["High", "Low", "Moderate"]  # must match training order

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------------
# Load the model once at startup and "warm it up" with a dummy prediction.
# The very first call to a Keras model on CPU is slow because TensorFlow
# builds its internal computation graph then — doing that now means your
# first real user never pays that cost.
# ----------------------------
_model = None


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
        dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
        _model(dummy, training=False)  # warm-up call, result discarded
    return _model


# ----------------------------
# User storage helpers
# ----------------------------
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def login_required(view):
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to continue.")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("predict"))
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.")
            return redirect(url_for("signup"))

        users = load_users()
        if email in users:
            flash("An account with that email already exists.")
            return redirect(url_for("signup"))

        users[email] = {
            "name": name,
            "password_hash": generate_password_hash(password),
        }
        save_users(users)

        session["user"] = {"name": name, "email": email}
        return redirect(url_for("predict"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        users = load_users()
        user = users.get(email)

        if not user or not check_password_hash(user["password_hash"], password):
            flash("Incorrect email or password.")
            return redirect(url_for("login"))

        session["user"] = {"name": user["name"], "email": email}
        return redirect(url_for("predict"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No image selected.")
            return redirect(url_for("predict"))

        file = request.files["image"]
        if file.filename == "" or not allowed_file(file.filename):
            flash("Please choose a valid image file (jpg, jpeg, png, bmp).")
            return redirect(url_for("predict"))

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        result = run_prediction(save_path)

        return render_template(
            "predict.html",
            user=session["user"],
            result=result,
            image_url=url_for("static", filename=f"uploads/{filename}"),
        )

    return render_template("predict.html", user=session["user"], result=None)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    """JSON endpoint, used by the page's fetch() call for an in-place update."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type."}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    result = run_prediction(save_path)
    result["image_url"] = url_for("static", filename=f"uploads/{filename}")
    return jsonify(result)


def run_prediction(image_path):
    model = get_model()

    img = keras_image.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Calling the model directly is much faster than model.predict() for a
    # single image — model.predict() rebuilds a tf.data pipeline every call.
    preds = model(img_array, training=False).numpy()[0]
    predicted_idx = int(np.argmax(preds))

    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": round(float(preds[predicted_idx]) * 100, 2),
        "probabilities": {
            cls: round(float(prob) * 100, 2)
            for cls, prob in zip(CLASS_NAMES, preds)
        },
    }


if __name__ == "__main__":
    print("Loading model...")
    get_model()  # load + warm up once, before serving any requests
    print("Model ready.")
    # use_reloader=False avoids starting two processes (and importing
    # TensorFlow twice) when debug=True
    app.run(debug=True, use_reloader=False)
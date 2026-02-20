import os
import numpy as np
import cv2
import base64
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------
# Initialize Flask App
# --------------------------------
app = Flask(__name__)

# --------------------------------
# Load Trained Model
# --------------------------------
model = load_model("Blood Cell.h5")

# IMPORTANT: Must match training class_indices
class_labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']


# --------------------------------
# Prediction Function
# --------------------------------
def predict_image_class(image_path):

    # Read image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to training size
    img_resized = cv2.resize(img_rgb, (224, 224))

    # VERY IMPORTANT: same preprocessing as training
    img_resized = preprocess_input(img_resized)

    # Add batch dimension
    img_input = np.expand_dims(img_resized, axis=0)

    # Predict
    prediction = model.predict(img_input)

    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_index]

    return predicted_label, img_rgb


# --------------------------------
# Home Route
# --------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        file = request.files["file"]

        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            predicted_label, img_rgb = predict_image_class(file_path)

            # Convert image to base64 for display
            _, buffer = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            img_str = base64.b64encode(buffer).decode("utf-8")

            return render_template(
                "result.html",
                class_label=predicted_label,
                img_data=img_str
            )

    return render_template("home.html")


# --------------------------------
# Run Server
# --------------------------------
if __name__ == "__main__":
    app.run(debug=True)
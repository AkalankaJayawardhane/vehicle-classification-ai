import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# =============================
# LOAD MODEL + LABELS
# =============================
model = tf.keras.models.load_model("best_vehicle10_model.h5")

with open("labels10.json", "r") as f:
    class_names = json.load(f)

IMG_SIZE = (160,160)

# =============================
# IMAGE UPLOAD
# =============================
def predict_uploaded_image():

    Tk().withdraw()
    file_path = askopenfilename(title="Select Vehicle Image")

    if not file_path:
        print("No image selected.")
        return

    # =============================
    # LOAD IMAGE
    # =============================
    img = image.load_img(file_path, target_size=IMG_SIZE)

    img_array = image.img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # =============================
    # PREDICT
    # =============================
    probabilities = model.predict(img_input, verbose=0)[0]
    predicted_class = np.argmax(probabilities)

    # =============================
    # SHOW RESULTS
    # =============================
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Uploaded Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.barh(class_names, probabilities * 100)
    plt.xlabel("Confidence (%)")
    plt.title(
        f"Predicted: {class_names[predicted_class]} "
        f"({probabilities[predicted_class]*100:.1f}%)"
    )
    plt.xlim(0,100)
    plt.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =============================
    # TOP 3
    # =============================
    print("\nTop 3 Predictions:")

    top_3_indices = np.argsort(probabilities)[-3:][::-1]

    for idx in top_3_indices:
        print(f"{class_names[idx]}: {probabilities[idx]*100:.2f}%")

# run
predict_uploaded_image()
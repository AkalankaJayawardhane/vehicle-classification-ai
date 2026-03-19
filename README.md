# 🚗 Vehicle Classification AI

## 📌 Project Overview

This project is a **Vehicle Classification System** developed using Deep Learning.
It can identify and classify different types of vehicles from an input image.

The model is trained to recognize **10 different vehicle categories** and provides predictions based on uploaded images.

---

## 🎯 Features

* Classifies 10 types of vehicles
* Uses a trained deep learning model (.h5)
* Predicts vehicle type from an input image
* Includes training and prediction scripts

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* JSON (for labels & training history)

---

## 📂 Project Structure

```
VehicleClassification/
│
├── Predict.py                 # Prediction script
├── Train.py                   # Training script
├── best_vehicle10_model.h5    # Trained model
├── labels10.json              # Class labels
├── training_history10.json    # Training results
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The dataset used for training is **not included in this repository** due to size limitations.

👉 Dataset Link: (Add your Google Drive / Kaggle link here)

---

## ▶️ How to Run

### 1. Install Requirements

```bash
pip install tensorflow numpy
```

### 2. Run Prediction

```bash
python Predict.py
```

---

## 📸 Example Output

Upload an image → Model predicts the vehicle category.

---

## 👨‍💻 Author

**Akalanka Jayawardhane**

---

## 📌 Note

This project was developed as part of an academic / personal AI project.

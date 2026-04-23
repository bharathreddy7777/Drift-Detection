# 🚀 Drift Detection System

A machine learning system to detect and handle **data drift** in real-time or batch environments.
This project monitors changes in data distribution and automatically adapts models to maintain performance.

---

## 📌 Features

* 📊 Detects **Covariate Drift** (feature distribution changes)
* 🔁 Detects **Concept Drift** (target relationship changes)
* ⚡ Supports **data stream processing**
* 📈 Performance evaluation and visualization
* 🤖 Automatic model retraining
* 🧪 Works with real datasets

---

## 🗂️ Project Structure

```
├── app.py                  # Main application
├── config.py               # Configuration settings
├── drift/                  # Drift detection modules
│   ├── concept_drift.py
│   ├── covariate_drift.py
│   ├── data_stream.py
│   └── drift_detectors.py
├── training/               # Model training & retraining
├── evaluation/             # Performance metrics & plots
├── data/                   # Dataset files
├── models/                 # Saved ML models
├── results/                # Output results
├── plots/                  # Visualization scripts
└── requirements.txt        # Dependencies
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/bharathreddy7777/Drift-Detection.git
cd Drift-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python app.py
```

---

## 📊 How It Works

1. Load dataset
2. Train baseline model
3. Monitor incoming data
4. Detect drift using statistical methods
5. Retrain model when drift is detected
6. Evaluate performance

---

## 📈 Output

* Drift detection results
* Model performance metrics
* Visualization plots
* CSV result files

---

## 🧠 Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

---

## 🎯 Use Cases

* Fraud detection systems
* Credit risk modeling
* Real-time ML monitoring
* Production ML systems

---

## 👨‍💻 Author

**Bharath Reddy**
GitHub: https://github.com/bharathreddy7777

---

## ⭐ Acknowledgement

This project is developed as part of a machine learning system to understand and handle **data drift in real-world applications**.


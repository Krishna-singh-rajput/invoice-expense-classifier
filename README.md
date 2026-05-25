# Invoice Expense Classifier API

## 📌 Overview
This project is a Machine Learning-based API that classifies invoice text into predefined expense categories.

The system takes invoice descriptions as input and predicts the appropriate category using a trained ML model.

---

## 🎯 Categories

- Logistics  
- Office Supplies  
- Cloud/Software  
- Utilities  
- Travel  
- Inventory  

---

## ⚙️ Tech Stack

- Python  
- FastAPI  
- Scikit-learn  
- TF-IDF Vectorizer  
- Logistic Regression  
- Joblib  

---

## 📂 Project Structure

```
invoice-expense-classifier/
│
├── data/
│   └── invoices.csv
├── model/
│   └── model.pkl
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Setup Instructions

### 1. Clone Repository

```
git clone <repo-link>
cd invoice-expense-classifier
```

---

### 2. Create Virtual Environment

```
python -m venv venv
```

Activate:

**Windows (PowerShell / CMD):**
```
venv\Scripts\activate
```

**Git Bash:**
```
source venv/Scripts/activate
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

### 4. Train Model

```
python train.py
```

This will generate:

```
model/model.pkl
```

---

### 5. Run API

```
uvicorn app:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Usage

### Endpoint:
```
POST /predict
```

### Request:
```json
{
  "text": "AWS monthly cloud hosting bill"
}
```

### Response:
```json
{
  "category": "Cloud/Software"
}
```

---

## 🧠 Model Details

- Text preprocessing: Lowercasing  
- Feature extraction: TF-IDF  
- Model: Logistic Regression  
- Training data: Custom dataset  

---

## ⚠️ Notes

- Ensure Python 3.10 or 3.11 is used  
- Do not include `venv/` in repository  
- Model is trained on a small dataset for demonstration  

---

## ✅ Features

- End-to-end ML pipeline  
- REST API using FastAPI  
- Real-time prediction  
- Easy to run locally  

---


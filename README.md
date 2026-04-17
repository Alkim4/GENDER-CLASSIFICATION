# Gender Classifier — Lab Activity 5 Part 3

Image classification web app using SVM with handcrafted features.  
**Subject:** ITIM 82 · 2nd Sem 2025–2026

---

## About

This app classifies an uploaded image as **Female** or **Male** using:

- **Model:** Support Vector Machine (RBF kernel, C=1)
- **Features:** 100 selected features via SelectKBest from 1,851 handcrafted features
- **Feature groups:** Color (RGB/HSV/Lab/Gray), GLCM, LBP, Shape, HOG, Lesion
- **Test Accuracy:** 85.10% | **Test F1:** 0.8513

---

## Project Structure

```
gender-classifier/
├── app.py               ← Streamlit web app
├── requirements.txt     ← Python dependencies
├── models/
│   ├── svm_model.pkl    ← Trained SVM model
│   ├── scaler.pkl       ← Fitted StandardScaler
│   ├── selector.pkl     ← SelectKBest selector
│   ├── label_encoder.pkl← LabelEncoder (Female/Male)
│   └── feature_names.pkl← List of 100 selected feature names
└── README.md
```

---

## Deployment Guide (Streamlit Community Cloud)

### Step 1 — Create a GitHub repository

1. Go to [github.com](https://github.com) → **New repository**
2. Name it `gender-classifier` (or any name you like)
3. Set it to **Public**
4. Click **Create repository**

### Step 2 — Upload files to GitHub

Upload all files maintaining this exact folder structure:

```
gender-classifier/          ← root of repo
├── app.py
├── requirements.txt
├── models/
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   ├── selector.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
└── README.md
```

> **Important:** The `models/` folder must be inside the repo root, not anywhere else.

### Step 3 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your **GitHub account**
3. Click **New app**
4. Fill in:
   - **Repository:** `your-username/gender-classifier`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy**
6. Wait ~2–3 minutes for it to build

Your app will be live at:  
`https://your-username-gender-classifier-app-XXXX.streamlit.app`

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## How It Works

1. User uploads an image (JPG, PNG, BMP, TIFF, WebP)
2. Image is resized to 256×256
3. All 1,851 handcrafted features are extracted
4. Features are scaled with `StandardScaler`
5. Top 100 features are selected by name
6. SVM model predicts: **Female** or **Male**

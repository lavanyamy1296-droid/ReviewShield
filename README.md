# ReviewShield AI — Fake Product Review Detector

A beginner-friendly machine-learning web app that screens product review text using TF-IDF + Logistic Regression and Random Forest.

## Features
- Single-review prediction
- Estimated model probability
- Bulk CSV analysis and CSV download
- Model metrics dashboard
- Bundled starter dataset; no API key or hardware required

## Requirements
- Python 3.10–3.12 recommended
- VS Code
- Internet connection for the first package installation

## Run on Windows PowerShell

Open the `ReviewShield-AI` folder in VS Code, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If `py -3.12` is unavailable, install Python 3.12 from python.org and enable **Add Python to PATH**. You can also use `python -m venv .venv` if `python --version` shows Python 3.10–3.12.

If PowerShell blocks activation, run this once in the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Streamlit prints a local address, normally `http://localhost:8501`. Open it in your browser.

## Dataset format
`dataset/reviews.csv` must contain:
- `review`: review text
- `label`: `fake` or `genuine`

To use your own dataset, keep those exact column names and labels.

## Important limitation
The included CSV is a small, curated starter dataset so the application runs immediately. It is for demonstrating the ML workflow, not for dependable real-world authenticity decisions. Text alone cannot prove whether a review is fake. For a stronger project, use a larger, independently labelled public dataset and evaluate on products/reviewers not seen during training.

## GitHub
Create a new empty repository, then from this folder:
```powershell
git init
git add .
git commit -m "Initial commit - ReviewShield AI"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ReviewShield-AI.git
git push -u origin main
```

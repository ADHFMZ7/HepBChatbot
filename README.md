# Hepatitis B Educational Chatbot

## 📌 Project Overview

This repository contains code and data used to build and evaluate the *Hepatitis B Educational Chatbot*.  
The goal of this project is to provide accurate, evidence-based information about Hepatitis B (HBV), and help combat misinformation through a simple, rule-based chatbot and associated evaluation tools.

⚠️ **Disclaimer:**  
This chatbot is for educational use only and does **not** diagnose or treat medical conditions.  
Users should seek professional medical advice for health decisions.

---
### 📦 Prerequisites

- Python **3.9+**
- Installed packages (use pip):

```bash
pip install -r requirements.txt

## 🚀 **How to Run the Program**

1) Download the program files (Download Zip)
2) Extract the ZIP locally
3) Open a terminal inside the extracted folder
4) Install packages uses pip (pip install -r requirements.txt)
5) Depending on your system, Python may be invoked as 'python', 'python3' or 'py'. Use the command that works with your machine

# Windows (most common)
py -m uvicorn HepBChat_refined:app --reload

# macOS / Linux
python3 -m uvicorn HepBChat_refined:app --reload

# Alternative (some systems)
python -m uvicorn HepBChat_refined:app --reload

6) Run the above code in terminal and server will start locally.
7) Go to http://127.0.0.1:8000 to access the chatbot
(For interactive API documentation, visit http://127.0.0.1:8000/docs)
8) Once finished with interacting with chatbot, press ctrl + c in terminal to end the program.

**Purpose of each file/folder**
HepBChatbot/
│
├── data/
│   └── Stores structured data files used by the chatbot and evaluation scripts.
│      (Does not include __pycache__.)
│
├── HepBChat_refined.py
│   └── Main application file.
│      Defines the FastAPI app, chatbot logic, rule-based NLU intent classifier,
│      response composition, and API endpoints.
│
├── intent_accuracy.csv
│   └── CSV file containing per-intent accuracy results generated during evaluation.
│
├── intent_accuracy.png
│   └── Visualization of intent-level classification accuracy.
│
├── final_accuracy_bar_graph.png
│   └── Final bar graph comparing overall chatbot performance results.
│
├── plot_accuracy.py
│   └── Script used to generate accuracy plots from evaluation CSV files.
│
├── test_accuracy.py
│   └── Script for testing the chatbot on a benchmark question set and computing
│      accuracy metrics.
│
├── requirements.txt
│   └── Lists all Python dependencies required to run the chatbot and evaluation scripts.
│







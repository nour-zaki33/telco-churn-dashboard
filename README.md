## Telco Customer Churn Analysis Dashboard

An interactive Streamlit app for exploring the Telco Customer Churn dataset, building simple churn prediction models (Logistic Regression and Random Forest), and simulating discount strategies to convert month-to-month customers to annual contracts.

### Project Structure
- `streamlit_app.py`: Main Streamlit app.
- `info.ipynb`: Exploration/notes for the dataset and preprocessing.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Dataset (Kaggle Telco Customer Churn).
- `archive.zip`: Archived resources (optional).

### Features
- Customer overview with churn distribution and contract-type insights
- Switchable models: Logistic Regression or Random Forest
- Metrics: accuracy, precision, recall, and confusion matrix plot
- Feature importance (for Random Forest)
- Discount strategy simulator to estimate conversion and net gain

### Prerequisites
- Python 3.9+ recommended
- Windows PowerShell or a terminal

### Install Dependencies
You can either install packages directly or create a virtual environment first (recommended).

1) Create and activate a virtual environment (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

2) Install required packages:
```powershell
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

### Dataset Path Note (Important)
The app reads the CSV with an absolute path. If your folder is different, update the path in `streamlit_app.py` inside the `load_data()` function:
```python
df = pd.read_csv(r"C:\Users\Finance\Desktop\telcom\WA_Fn-UseC_-Telco-Customer-Churn.csv")
```
If needed, replace it with a relative path (recommended):
```python
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```
Ensure the CSV file is located in the project root alongside `streamlit_app.py`.

### Run the App
From the project directory:
```powershell
streamlit run streamlit_app.py
```
Then open the URL shown in the terminal (usually http://localhost:8501).

### Troubleshooting
- File not found: Verify the CSV exists and the path in `streamlit_app.py` matches your location. Prefer using a relative path as shown above.
- Import errors: Re-run `pip install` for the listed packages and ensure your virtual environment is activated.
- Port already in use: Run with a different port, e.g. `streamlit run streamlit_app.py --server.port 8502`.

### Optional: Freeze Requirements
If you want to capture exact versions used locally:
```powershell
pip freeze > requirements.txt
```
Others can then install with:
```powershell
pip install -r requirements.txt
```

### License
For personal/educational use. Replace with your preferred license if sharing publicly.



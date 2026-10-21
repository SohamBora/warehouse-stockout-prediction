# Warehouse Stockout Risk Prediction & Recommendation System

A Python, scikit-learn, and Streamlit application for classifying warehouse products into **Low**, **Medium**, or **High** stockout-risk categories.

## Features
- Upload a warehouse inventory CSV through the Streamlit sidebar.
- Validate required columns and clean rows with missing values and duplicates.
- Train a Logistic Regression classifier using selected inventory and sales features.
- Display stockout-risk predictions and model evaluation information in the dashboard.
- Provide inventory recommendations through the app's recommendation workflow.

## Tech stack
- Python
- Pandas and NumPy
- scikit-learn
- Streamlit

## Setup and run locally
1. Install Python 3.10 or newer.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   ```
3. Install dependencies: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`
5. In the app sidebar, upload `data/warehouse_stockout_unclean_2000.csv`.

## Input data
The included CSV is an uncleaned practice dataset. Required columns include `Current_Stock`, `Average_Daily_Sales`, `Sales_Last_7_Days`, `Sales_Last_30_Days`, `Supplier_Lead_Time_Days`, `Reorder_Level`, and `Stockout_Risk`. The target label has Low, Medium, and High classes.

## Screenshot
`screenshots/dashboard-preview.png` is a visual preview of the app layout, not a screenshot captured from a live running session. Replace it with a real screenshot after launching the app locally.

## Notes
- No accuracy or F1-score is claimed here; report only metrics reproduced from the current code's evaluation output.
- The model is retrained when a CSV is uploaded; no serialized model is loaded.
- Review code and data for private or licensed information before publishing.

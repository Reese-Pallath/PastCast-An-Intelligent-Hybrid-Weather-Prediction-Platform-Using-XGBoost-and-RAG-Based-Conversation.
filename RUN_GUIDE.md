# PastCast: Local Run Instructions

This guide explains how to spin up the entire application locally on your computer for demonstration.

Open two separate terminal windows on your Mac:
##Terminal 1 — Run the Backend (Flask + ML Models):##
bash
# 1. Navigate to the backend directory
cd backend

# 2. Activate the virtual environment
 source ../.venv/bin/activate. or source .venv/bin/activate.

# 3. Install dependencies (if you haven't yet)
pip install -r requirements.txt

#and *(On Mac, you may also need to run `brew install libomp` for XGBoost to work).*

# 4. Start the backend app server
python app.py
Note: The backend will be running on http://localhost:8000.

##Terminal 2 — Run the Frontend (React):

# 1. Navigate to your project root
cd /Users/reesepallath/Documents/PastCast_Backups/pastcast-Main

# 2. Install any missing Node/React packages
npm install

# 3. Start the React development server
npm start
Note: The frontend will open in your browser automatically at http://localhost:3000.


### 3. Start the MLflow Dashboard
Open a third terminal, navigate to the `backend/` folder, and launch the MLflow tracking web interface:
```bash
cd backend
../.venv/bin/mlflow ui --backend-store-uri file://$(pwd)/ml/mlruns --port 5001
```
You can view your logged XGBoost machine learning metrics by opening your browser to **http://127.0.0.1:5001**

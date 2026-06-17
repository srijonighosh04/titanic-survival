# Titanic Survival Prediction Pipeline 🚢

This repository contains a self-contained, end-to-end Machine Learning pipeline implemented in Python to predict passenger survival on the Titanic using **Logistic Regression**. 

The pipeline automatically downloads the raw Titanic dataset, handles missing data, preprocesses features, trains a classification model, evaluates it, and generates analysis visualizations.

---

## 📁 Project Structure

```text
titanic-survival/
├── data/
│   └── titanic.csv             # Cached raw dataset
├── output/
│   ├── confusion_matrix.png    # Visualization of model classification performance
│   └── feature_importance.png  # Horizontal bar plot showing feature coefficients
├── main.py                     # Main Python script executing the pipeline
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation (this file)
```

---

## 🛠️ Installation & Setup

To run this pipeline locally, follow these steps:

### 1. Set Up a Virtual Environment (Recommended)
Creating a virtual environment ensures dependencies do not conflict with other Python installations on your system.

**On Windows (PowerShell/CMD):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install all required libraries listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Execute the entire pipeline by running [main.py](file:///c:/Users/SRIJONI%20GHOSH/Documents/Coding/titanic-survival/main.py):
```bash
python main.py
```

Running the pipeline executes the following stages sequentially:
1. **Download**: Downloads the Titanic dataset if it is not already present in the `data/` directory.
2. **Preprocess**: Cleans the data, handles missing values, and encodes categorical columns.
3. **Train**: Splits the dataset into 80% training / 20% test sets, scales the variables, and trains a Logistic Regression classifier.
4. **Evaluate**: Evaluates accuracy and outputs a detailed classification report (precision, recall, f1-score) in the terminal.
5. **Visualize**: Generates and saves analytics plots to the `output/` directory.

---

## ⚙️ How the Pipeline Works

### 1. Data Ingestion (`download_data`)
Retrieves the dataset from the Data Science Dojo raw repository and caches it locally under `data/titanic.csv`.

### 2. Preprocessing (`preprocess_data`)
- **Missing Value Imputation**:
  - `Age` is imputed with its median value.
  - `Embarked` is imputed with its mode (most frequent port).
- **Feature Selection**:
  - Drops columns not helpful for classification or containing too many nulls: `PassengerId`, `Name`, `Ticket`, and `Cabin`.
- **Categorical Encoding**:
  - Converts `Sex` and `Embarked` to dummy/indicator variables via one-hot encoding, using `drop_first=True` to avoid the dummy variable trap.

### 3. Model Training & Evaluation (`train_and_evaluate`)
- Splits data into train and test sets using a fixed seed (`random_state=42`) for reproducibility.
- Applies standard scaling (`StandardScaler`) to features to ensure scale uniformity.
- Trains a `LogisticRegression` classifier and outputs evaluation metrics.

### 4. Visualizations (`plot_results`)
The pipeline generates and saves:
- **`confusion_matrix.png`**: Displays true positives, true negatives, false positives, and false negatives.
- **`feature_importance.png`**: Visualizes the learned regression coefficients (log-odds), showing which features contribute most strongly to prediction outcome (e.g., negative coefficients indicating reduced survival probability).

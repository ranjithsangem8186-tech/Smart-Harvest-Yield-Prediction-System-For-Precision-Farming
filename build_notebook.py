import json
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def build_notebook_json():
    # Define notebook cells
    cells = []
    
    # Cell 1: Markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🌾 Smart Harvest Yield Prediction for Precision Farming\n",
            "### Academic Final Year Project - Model Development Workflow\n",
            "\n",
            "This notebook details the full machine learning engineering workflow for predicting agricultural crop yield based on environmental weather features, soil health attributes, and N-P-K nutrient parameters. \n",
            "\n",
            "---\n",
            "\n",
            "### 🧠 Precision Agriculture Overview\n",
            "Precision Farming (or Precision Agriculture) uses technology-driven insights to optimize field-level management, particularly with regard to crop cultivation. By predicting crop yields dynamically using soil characteristics and real-time/expected weather forecasts, farmers can:\n",
            "1. **Select the best crop** for their regional climate and soil type.\n",
            "2. **Optimize nutrient inputs** (Nitrogen, Phosphorus, Potassium) to reduce costs and avoid chemical runoff.\n",
            "3. **Improve yields** by anticipating weather deviations and planning harvests accordingly.\n",
            "\n",
            "---\n",
            "\n",
            "### ⚙️ Machine Learning Workflow:\n",
            "- **Data Loading & Inspections**: Ingest agricultural survey records.\n",
            "- **Exploratory Data Analysis (EDA)**: Render 5 crop-wise and environmental charts.\n",
            "- **Feature Engineering**: Create agronomic ratios (total N-P-K, temperature/humidity ratios).\n",
            "- **Preprocessing**: Encode labels for states, districts, crops, and seasons. Scale all values.\n",
            "- **Model Training & Evaluation**: Train a Random Forest Regressor and report MAE, RMSE, and $R^2$ scores.\n",
            "- **Serialization**: Export the trained model, label encoders, features list, and scaler to `.joblib` files."
        ]
    })
    
    # Cell 2: Code (Imports)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import joblib\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.ensemble import RandomForestRegressor\n",
            "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
            "import os\n",
            "\n",
            "# Set visualization settings\n",
            "sns.set_theme(style=\"whitegrid\")\n",
            "plt.rcParams['figure.figsize'] = (10, 6)\n",
            "print(\"Libraries successfully imported!\")"
        ]
    })

    # Cell 3: Code (Data Loading)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load the dataset\n",
            "dataset_path = \"dataset.csv\"\n",
            "if not os.path.exists(dataset_path):\n",
            "    print(\"dataset.csv not found! Run the dataset generator first.\")\n",
            "else:\n",
            "    df = pd.read_csv(dataset_path)\n",
            "    print(f\"Dataset Loaded successfully! Shape: {df.shape}\")\n",
            "    print(\"\\n--- First 5 Columns of Crop Yield Records ---\")\n",
            "    display(df.head())\n",
            "    print(\"\\n--- Descriptive Summary ---\")\n",
            "    display(df.describe())"
        ]
    })

    # Cell 4: Markdown (Data Cleaning)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 🧹 Data Cleaning and Missing Value Treatment\n",
            "Real agriculture datasets have missing records and outliers due to sensor errors or recording failures. We inspect missing cells, duplicates, and clean the data before model fitting."
        ]
    })

    # Cell 5: Code (Cleaning)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Check missing values\n",
            "print(\"Missing values per column:\")\n",
            "print(df.isnull().sum())\n",
            "\n",
            "# Check duplicates\n",
            "duplicates = df.duplicated().sum()\n",
            "print(f\"\\nDuplicate rows found: {duplicates}\")\n",
            "if duplicates > 0:\n",
            "    df = df.drop_duplicates()\n",
            "    print(\"Duplicate rows dropped!\")\n",
            "\n",
            "print(\"\\nUnique categorical factors:\")\n",
            "for col in ['state', 'district', 'crop', 'season', 'soil_type']:\n",
            "    print(f\"- {col}: {df[col].nunique()} unique classes\")"
        ]
    })

    # Cell 6: Markdown (EDA Introduction)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 📊 Exploratory Data Analysis (EDA)\n",
            "Let's create academic-grade visualizations to inspect the feature distributions and trends."
        ]
    })

    # Cell 7: Code (EDA 1: Crop-wise Yield)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10, 5))\n",
            "crop_yield = df.groupby('crop')['yield'].mean().sort_values(ascending=False)\n",
            "sns.barplot(x=crop_yield.values, y=crop_yield.index, palette=\"viridis\")\n",
            "plt.title(\"Average Yield of Indian Crops (Tons/Hectare)\", fontsize=14, fontweight='bold', color='#1E3F20')\n",
            "plt.xlabel(\"Yield (Tons/Ha)\", fontsize=12)\n",
            "plt.ylabel(\"Crop Type\", fontsize=12)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # Cell 8: Code (EDA 2: State-wise Yield)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10, 5))\n",
            "state_yield = df.groupby('state')['yield'].mean().sort_values(ascending=False)\n",
            "sns.barplot(x=state_yield.values, y=state_yield.index, palette=\"crest\")\n",
            "plt.title(\"Average Crop Productivity across Indian States\", fontsize=14, fontweight='bold', color='#1E3F20')\n",
            "plt.xlabel(\"Yield (Tons/Ha)\", fontsize=12)\n",
            "plt.ylabel(\"Indian State\", fontsize=12)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # Cell 9: Code (EDA 3: Rainfall vs Yield)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10, 6))\n",
            "# Visualizing contrasting crops: Rice (high moisture preferred) vs Wheat (low moisture preferred)\n",
            "sub_df = df[df['crop'].isin(['Rice', 'Wheat'])]\n",
            "sns.scatterplot(data=sub_df, x='rainfall', y='yield', hue='crop', alpha=0.7, palette=['#4A7C59', '#D4A373'])\n",
            "plt.title(\"Rainfall vs. Yield Response (Rice vs. Wheat)\", fontsize=14, fontweight='bold', color='#1E3F20')\n",
            "plt.xlabel(\"Rainfall (mm)\", fontsize=12)\n",
            "plt.ylabel(\"Yield (Tons/Ha)\", fontsize=12)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # Cell 10: Code (EDA 4: Temp vs Yield)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10, 6))\n",
            "sns.scatterplot(data=df, x='temperature', y='yield', hue='season', alpha=0.6, palette=\"coolwarm\")\n",
            "plt.title(\"Temperature vs. Yield Distribution across Seasons\", fontsize=14, fontweight='bold', color='#1E3F20')\n",
            "plt.xlabel(\"Temperature (°C)\", fontsize=12)\n",
            "plt.ylabel(\"Yield (Tons/Ha)\", fontsize=12)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # Cell 11: Code (EDA 5: Soil Suitability)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10, 6))\n",
            "sns.boxplot(data=df, x='soil_type', y='yield', palette=\"Set2\")\n",
            "plt.title(\"Yield Variation by Soil Texture Type\", fontsize=14, fontweight='bold', color='#1E3F20')\n",
            "plt.xlabel(\"Soil Classification\", fontsize=12)\n",
            "plt.ylabel(\"Yield (Tons/Ha)\", fontsize=12)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # Cell 12: Code (EDA 6: Correlation matrix)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10, 8))\n",
            "numeric_cols = df.select_dtypes(include=[np.number]).columns\n",
            "correlation_matrix = df[numeric_cols].corr()\n",
            "sns.heatmap(correlation_matrix, annot=True, cmap=\"YlGnBu\", fmt=\".2f\", linewidths=0.5)\n",
            "plt.title(\"Nutrient & Weather Correlation Matrix\", fontsize=14, fontweight='bold', color='#1E3F20')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # Cell 13: Markdown (Feature Engineering)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### ⚙️ Feature Engineering & Preprocessing\n",
            "We enrich the feature space by creating agronomic features:\n",
            "1. **`NPK_Total`**: Combined N, P, K quantity in the soil (total chemical inputs index).\n",
            "2. **`temp_humidity_ratio`**: Ratio indicating evapotranspiration rates.\n",
            "3. **`rain_temp_ratio`**: Ratio of moisture index."
        ]
    })

    # Cell 14: Code (Feature Engineering Code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create agronomic feature combinations\n",
            "df['NPK_Total'] = df['nitrogen'] + df['phosphorus'] + df['potassium']\n",
            "df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)\n",
            "df['rain_temp_ratio'] = df['rainfall'] / (df['temperature'] + 1)\n",
            "\n",
            "print(\"Engineered Features Created! Dimensions:\", df.shape)\n",
            "display(df[['nitrogen', 'phosphorus', 'potassium', 'NPK_Total', 'temp_humidity_ratio', 'rain_temp_ratio']].head())"
        ]
    })

    # Cell 15: Markdown (Preprocessing intro)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 🔠 Encoding Categoricals & Standard Scaling\n",
            "- Categorical strings (`state`, `district`, `crop`, `season`, `soil_type`) are encoded into index categories using Scikit-Learn's `LabelEncoder`.\n",
            "- Numerical values are scaled using `StandardScaler` to speed up and stabilize model training coefficients."
        ]
    })

    # Cell 16: Code (Encoding and Scaling execution)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "cat_cols = ['state', 'district', 'crop', 'season', 'soil_type']\n",
            "label_encoders = {}\n",
            "df_encoded = df.copy()\n",
            "\n",
            "for col in cat_cols:\n",
            "    le = LabelEncoder()\n",
            "    df_encoded[col] = le.fit_transform(df[col])\n",
            "    label_encoders[col] = le\n",
            "    print(f\"Encoded '{col}' successfully.\")\n",
            "\n",
            "# Split features X and target label y\n",
            "X = df_encoded.drop(columns=['yield'])\n",
            "y = df_encoded['yield']\n",
            "feature_columns = list(X.columns)\n",
            "\n",
            "# Standardize numeric inputs\n",
            "scaler = StandardScaler()\n",
            "X_scaled = scaler.fit_transform(X)\n",
            "print(\"\\nAll categorical inputs encoded, numeric features scaled.\")\n",
            "print(f\"Final X shape: {X_scaled.shape}\")"
        ]
    })

    # Cell 17: Markdown (Model fitting)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 🤖 Random Forest Regressor Model Training\n",
            "We divide the dataset into an **80-20 train-test split** and fit a **Random Forest Regressor** model."
        ]
    })

    # Cell 18: Code (Model Training)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
            "print(f\"Training set: {X_train.shape[0]} samples\")\n",
            "print(f\"Testing set: {X_test.shape[0]} samples\")\n",
            "\n",
            "# Initialize model\n",
            "rf_model = RandomForestRegressor(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1)\n",
            "print(\"Fitting Random Forest Model...\")\n",
            "rf_model.fit(X_train, y_train)\n",
            "print(\"Model training finished successfully!\")"
        ]
    })

    # Cell 19: Markdown (Evaluation intro)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 📈 Model Evaluation Metrics\n",
            "We inspect the Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$ Score) to gauge prediction accuracy."
        ]
    })

    # Cell 20: Code (Evaluation metrics calculations)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Perform prediction\n",
            "y_pred = rf_model.predict(X_test)\n",
            "\n",
            "# Compute metrics\n",
            "mae = mean_absolute_error(y_test, y_pred)\n",
            "rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
            "r2 = r2_score(y_test, y_pred)\n",
            "\n",
            "print(\"--- Model Evaluation Performance Summary ---\")\n",
            "print(f\"Mean Absolute Error (MAE)   : {mae:.4f} Tons/Ha\")\n",
            "print(f\"Root Mean Squared Error (RMSE) : {rmse:.4f} Tons/Ha\")\n",
            "print(f\"R² Score (Model Fit Accuracy) : {r2:.4f} ({r2*100:.2f}%)\")"
        ]
    })

    # Cell 21: Code (Actual vs Predicted chart)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(7, 7))\n",
            "plt.scatter(y_test, y_pred, alpha=0.5, color='#4A7C59', edgecolors='k')\n",
            "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2.5)\n",
            "plt.title(\"Yield Accuracy: Actual vs. Predicted\", fontsize=14, fontweight='bold', color='#1E3F20')\n",
            "plt.xlabel(\"Actual Crop Yield (Tons/Ha)\", fontsize=12)\n",
            "plt.ylabel(\"Predicted Crop Yield (Tons/Ha)\", fontsize=12)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # Cell 22: Code (Feature Importance chart)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(10, 6))\n",
            "importances = rf_model.feature_importances_\n",
            "indices = np.argsort(importances)[::-1]\n",
            "sorted_features = [feature_columns[i] for i in indices]\n",
            "\n",
            "sns.barplot(x=importances[indices], y=sorted_features, palette=\"summer\")\n",
            "plt.title(\"Random Forest Feature Importances for Yield Forecasts\", fontsize=14, fontweight='bold', color='#1E3F20')\n",
            "plt.xlabel(\"Attribute Contribution Score\", fontsize=12)\n",
            "plt.ylabel(\"Agronomic Feature\", fontsize=12)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # Cell 23: Markdown (Serialization intro)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 💾 Model Serialization\n",
            "To build our interactive farming dashboard, we write our trained model, label encoders, standard scaler, and columns index to the disk using `joblib` files."
        ]
    })

    # Cell 24: Code (Serialization execution)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Export joblib artifacts\n",
            "joblib.dump(rf_model, \"yield_prediction_model.joblib\")\n",
            "joblib.dump(label_encoders, \"label_encoders.joblib\")\n",
            "joblib.dump(scaler, \"scaler.joblib\")\n",
            "joblib.dump(feature_columns, \"feature_columns.joblib\")\n",
            "\n",
            "print(\"All ML artifacts exported successfully:\")\n",
            "print(\"- yield_prediction_model.joblib\")\n",
            "print(\"- label_encoders.joblib\")\n",
            "print(\"- scaler.joblib\")\n",
            "print(\"- feature_columns.joblib\")"
        ]
    })

    # Cell 25: Markdown (Academic Q&A)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 🧠 Academic Viva Questions & Quick-Reference Answers\n",
            "\n",
            "**Q1. What is the objective of this project?**  \n",
            "*Answer*: To forecast crop yields (Tons/Hectare) based on weather factors (rainfall, temperature, humidity), soil health variables (soil type, pH), and primary soil nutrients (Nitrogen, Phosphorus, Potassium). This supports precision farming by giving farmers quantitative insights to optimize resources.\n",
            "\n",
            "**Q2. Why did you choose Random Forest Regressor over simple Linear Regression?**  \n",
            "*Answer*: Crop yield forms a highly complex non-linear relationship with environmental inputs. For instance, too little or too much rainfall decreases yield. Linear regression cannot capture these complex peaks and optimal thresholds, whereas Random Forest, as an ensemble decision-tree method, excels at capturing high-dimensional non-linear interactions.\n",
            "\n",
            "**Q3. How did you handle categorical attributes like State, District, and Crop?**  \n",
            "*Answer*: We utilized Scikit-Learn's `LabelEncoder` to translate category strings into unique integer tokens. These encoders were saved to disk and are loaded dynamically by the Tkinter desktop GUI to process dropdown selections.\n",
            "\n",
            "**Q4. What is the role of feature scaling (StandardScaler)?**  \n",
            "*Answer*: It centers and scales each numerical column to have a mean of 0 and standard deviation of 1. This ensures that features with larger numeric scales (e.g., Rainfall in thousands of mm) do not disproportionately bias the learning algorithm over smaller scales (e.g., pH)."
        ]
    })
    
    # Notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # Write to notebook.ipynb
    with open("notebook.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print("Successfully built notebook.ipynb structure!")

def train_and_save_model():
    print("Loading generated dataset...")
    df = pd.read_csv("dataset.csv")
    
    # Feature engineering
    df['NPK_Total'] = df['nitrogen'] + df['phosphorus'] + df['potassium']
    df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)
    df['rain_temp_ratio'] = df['rainfall'] / (df['temperature'] + 1)
    
    # Encoding
    cat_cols = ['state', 'district', 'crop', 'season', 'soil_type']
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    X = df_encoded.drop(columns=['yield'])
    y = df_encoded['yield']
    feature_columns = list(X.columns)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1)
    print("Fitting model to generate joblib files...")
    rf_model.fit(X_train, y_train)
    
    # Save artifacts
    joblib.dump(rf_model, "yield_prediction_model.joblib")
    joblib.dump(label_encoders, "label_encoders.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(feature_columns, "feature_columns.joblib")
    
    # Metrics
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print("\nTraining Metrics:")
    print(f"- MAE: {mae:.4f} Tons/Ha")
    print(f"- RMSE: {rmse:.4f} Tons/Ha")
    print(f"- R2: {r2:.4f} ({r2*100:.2f}%)")
    print("Model joblib files successfully created!")

if __name__ == "__main__":
    build_notebook_json()
    train_and_save_model()

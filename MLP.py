# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 10:38:51 2024

@author: vertt
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import time

# Load and preprocess dataset
data = pd.read_excel("C:/Users/vertt/Desktop/Gradudata1.xlsx")
data = data.drop(columns=['CN', 'N', 'S'])

# Filter dataset to include only data up to 2019
data = data[data['YR'] <= 2019]

# Define dependent variables and their lags
y_variables_and_lags = {
    'Y_CFO': ['CFO', 'CFO_lag1'],
    'Y_FCF': ['FCF', 'FCF_lag1'],
    'Y_ROE': ['ROE', 'ROE_lag1'],
    'Y_ROA': ['ROA', 'ROA_lag1']}

# Economic variables and their lags
economic_vars = ['GDP', '3_month', '12_month']
economic_vars_lags = ['GDP_lagged', '3_month_l1', '12_month_l1']


# Prepare for saving results
metrics_results = []
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

start = time.time()

def sliding_window_validation(data, y_var, features, param_grid, train_years=7, test_years=1):
    """
    Perform walk-forward validation with a fixed-size sliding window for training and testing.
    """
    results = []
    best_params = None
    years = sorted(data['YR'].unique())
    total_slides = 5  # Number of sliding windows

    # Calculate the step size to slide the window 5 times
    if len(years) > train_years + test_years:
        step_years = max((len(years) - (train_years + test_years)) // (total_slides - 1), 1)
    else:
        raise ValueError("Insufficient data years to create the required sliding windows.")

    for start_idx in range(0, len(years) - (train_years + test_years) + 1, step_years):
        # Define training and testing year ranges
        train_year_range = years[start_idx : start_idx + train_years]
        test_year_range = years[start_idx + train_years : start_idx + train_years + test_years]

        print(f"Train years: {train_year_range}, Test year: {test_year_range}")

        train_data = data[data['YR'].isin(train_year_range)]
        test_data = data[data['YR'].isin(test_year_range)]

        if train_data.empty or test_data.empty:
            continue
        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        X_train = train_data[features]
        y_train = train_data[y_var]
        X_test = test_data[features]
        y_test = test_data[y_var]
        
        print(f"Classes in y_train: {y_train.value_counts().to_dict()}, Classes in y_test: {y_test.value_counts().to_dict()}")

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Parameter Optimization using AUC
        best_model = None
        best_auc = 0
        for params in ParameterGrid(param_grid):
            try:
                model = MLPClassifier(**params, random_state=42, max_iter=2500, early_stopping=True)
                model.fit(X_train_scaled, y_train)
                y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
                train_auc = roc_auc_score(y_train, y_train_proba)

                if train_auc > best_auc:
                    best_auc = train_auc
                    best_model = model
                    best_params = params
            except Exception:
                continue

        # Evaluate the best model on the test set
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

        results.append({
            'Train Years': train_year_range,
            'Test Year': test_year_range,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'Log Loss': log_loss(y_test, y_pred_proba),
            'AUC': roc_auc_score(y_test, y_pred_proba)
        })

    # Compute averages and variances for metrics
    metrics_summary = {
        'Accuracy Mean': np.mean([r['Accuracy'] for r in results]),
        'Accuracy Variance': np.var([r['Accuracy'] for r in results]),
        'Precision Mean': np.mean([r['Precision'] for r in results]),
        'Precision Variance': np.var([r['Precision'] for r in results]),
        'Recall Mean': np.mean([r['Recall'] for r in results]),
        'Recall Variance': np.var([r['Recall'] for r in results]),
        'F1 Mean': np.mean([r['F1 Score'] for r in results]),
        'F1 Variance': np.var([r['F1 Score'] for r in results]),
        'Log Loss Mean': np.mean([r['Log Loss'] for r in results]),
        'Log Loss Variance': np.var([r['Log Loss'] for r in results]),
        'AUC Mean': np.mean([r['AUC'] for r in results]),
        'AUC Variance': np.var([r['AUC'] for r in results])
    }

    return metrics_summary, best_params


# Loop through each target variable and run the different combinations
for y_var, features in y_variables_and_lags.items():
    for target_lag_option in ['without_lag', 'with_lag']:
        for econ_lag_option in ['no_econ', 'with_econ', 'with_econ_lag']:
            print(f"Running model for {y_var} ({target_lag_option}) with economic option: {econ_lag_option}")

            # Start with independent variables only
            independent_vars = [
                'YR', 'AR', 'AFA', 'COGS', 'CR', 'DR', 'IDAP', 'OR', 'NI', 'SF', 'TA', 'TL', 'ATA', 'N_coded', 'S_coded'
            ]
            X_current = data[independent_vars].copy()

            # Add economic variables based on the econ_lag_option
            if econ_lag_option == 'no_econ':
                X_current = X_current.drop(
                    columns=[col for col in economic_vars + economic_vars_lags if col in X_current.columns], 
                    errors='ignore'
                )
            elif econ_lag_option == 'with_econ':
                X_current = X_current.drop(
                    columns=[col for col in economic_vars_lags if col in X_current.columns], 
                    errors='ignore'
                )
                # Explicitly include base economic variables
                for econ_var in economic_vars:
                    if econ_var not in X_current.columns:
                        X_current[econ_var] = data[econ_var]
            elif econ_lag_option == 'with_econ_lag':
                # Include both economic variables and their lags
                for econ_var in economic_vars + economic_vars_lags:
                    if econ_var not in X_current.columns:
                        X_current[econ_var] = data[econ_var]
            
            # Ensure the target variable's baseline and lag are included
            if target_lag_option == 'without_lag':
                if features[0] not in X_current.columns:
                    X_current[features[0]] = data[features[0]]
            elif target_lag_option == 'with_lag':
                for feature in features:
                    if feature not in X_current.columns:
                      X_current[feature] = data[feature]
            # Debugging: Print selected features
            print(f"Selected features for econ_lag_option='{econ_lag_option}': {X_current.columns.tolist()}")

            # Define parameter grid
            param_grid = {
                'activation':['relu','tanh'],
                'hidden_layer_sizes': [(64, 32), (128, 64), (32, 16)],
                'alpha': [0.001, 0.01],  # Regularization parameter
                'learning_rate_init': [0.001, 0.01, 0.1],
                'learning_rate':['constant']
            }

            # Perform sliding window validation
            metrics_summary, best_params = sliding_window_validation(
                data, y_var, X_current.columns.tolist(), param_grid, train_years=7, test_years=1
            )

            # Save results
            metrics_results.append({
                'Target': y_var,
                'Target Lag Option': target_lag_option,
                'Economic Lag Option': econ_lag_option,
                'Best Parameters': str(best_params),  # Include best_params
                **metrics_summary
            })

# Save metrics to Excel
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_excel("C:/Users/vertt/Desktop/walk_forward_mlp_metrics.xlsx", index=False)

end = time.time()
print(f"Completed processing in {(end - start) / 60:.2f} minutes")

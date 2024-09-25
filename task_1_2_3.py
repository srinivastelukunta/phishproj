import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ydata_profiling import ProfileReport
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import shap
import joblib


# Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    print(f"--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.title(f'Confusion Matrix for {model_name}')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.show()

def task_1_2():

    ####################################################  TASK 1  #####################################################
    # Load the dataset
    df = pd.read_csv('Phishing_Legitimate_full.csv')  

    # Display the first few rows
    df.head()
    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing Values:\n", missing_values)

    # Since there are no missing values in this dataset, we proceed to check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")

    # Remove duplicate rows if any
    df = df.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {df.shape}")

    # Encode the target variable 'CLASS_LABEL'
    label_encoder = LabelEncoder()
    df['CLASS_LABEL'] = label_encoder.fit_transform(df['CLASS_LABEL'])


    # Separate features and target
    X = df.drop('CLASS_LABEL', axis=1)
    y = df['CLASS_LABEL']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the testing data
    X_test_scaled = scaler.transform(X_test)

    ####################################################  TASK 2  #####################################################
    # Initialize Logistic Regression
    log_reg = LogisticRegression(random_state=42)

    # Define hyperparameters for tuning
    log_reg_params = {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }

    # Initialize GridSearchCV
    log_reg_grid = GridSearchCV(
        estimator=log_reg,
        param_grid=log_reg_params,
        cv=5,
        scoring='accuracy',
    )

    # Fit the model
    log_reg_grid.fit(X_train_scaled, y_train)

    # Best parameters
    print("Best parameters for Logistic Regression:", log_reg_grid.best_params_)
    # Initialize Random Forest Classifier
    rf_clf = RandomForestClassifier(random_state=42)

    # Define hyperparameters for tuning
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    # Initialize GridSearchCV
    rf_grid = GridSearchCV(
        estimator=rf_clf,
        param_grid=rf_params,
        cv=5,
        scoring='accuracy',
    )

    # Fit the model
    rf_grid.fit(X_train_scaled, y_train)

    # Best parameters
    print("Best parameters for Random Forest:", rf_grid.best_params_)

        # Initialize Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier(random_state=42)

    # Define hyperparameters for tuning
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Initialize GridSearchCV
    gb_grid = GridSearchCV(
        estimator=gb_clf,
        param_grid=gb_params,
        cv=5,
        scoring='accuracy',
    )

    # Fit the model
    gb_grid.fit(X_train_scaled, y_train)

    # Best parameters
    print("Best parameters for Gradient Boosting:", gb_grid.best_params_)

    # Initialize SVM
    svm_clf = SVC(random_state=42, probability=True)

    # Define hyperparameters for tuning
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    # Initialize GridSearchCV
    svm_grid = GridSearchCV(
        estimator=svm_clf,
        param_grid=svm_params,
        cv=5,
        scoring='accuracy',
    )

    # Fit the model
    svm_grid.fit(X_train_scaled, y_train)

    # Best parameters
    print("Best parameters for SVM:", svm_grid.best_params_)

    # Evaluate Logistic Regression
    evaluate_model(log_reg_grid.best_estimator_, X_test_scaled, y_test, "Logistic Regression")

    # Evaluate Random Forest
    evaluate_model(rf_grid.best_estimator_, X_test_scaled, y_test, "Random Forest")

    # Evaluate Gradient Boosting
    evaluate_model(gb_grid.best_estimator_, X_test_scaled, y_test, "Gradient Boosting")

    # Evaluate SVM
    evaluate_model(svm_grid.best_estimator_, X_test_scaled, y_test, "Support Vector Machine")    

    # Compile results
    results = {
        'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM'],
        'Accuracy': [
            accuracy_score(y_test, log_reg_grid.best_estimator_.predict(X_test_scaled)),
            accuracy_score(y_test, rf_grid.best_estimator_.predict(X_test_scaled)),
            accuracy_score(y_test, gb_grid.best_estimator_.predict(X_test_scaled)),
            accuracy_score(y_test, svm_grid.best_estimator_.predict(X_test_scaled))
        ],
        'ROC_AUC': [
            roc_auc_score(y_test, log_reg_grid.best_estimator_.predict_proba(X_test_scaled)[:,1]),
            roc_auc_score(y_test, rf_grid.best_estimator_.predict_proba(X_test_scaled)[:,1]),
            roc_auc_score(y_test, gb_grid.best_estimator_.predict_proba(X_test_scaled)[:,1]),
            roc_auc_score(y_test, svm_grid.best_estimator_.predict_proba(X_test_scaled)[:,1])
        ]
    }

    results_df = pd.DataFrame(results)
    print(results_df)

def task_3():

    # Load the dataset
    df = pd.read_csv('Phishing_Legitimate_full.csv')  
    # Handle any missing values if present
    df = df.dropna()

    # Encode the target variable 'CLASS_LABEL'
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df['CLASS_LABEL'] = label_encoder.fit_transform(df['CLASS_LABEL'])

    # Separate features and target
    X = df.drop('CLASS_LABEL', axis=1)
    y = df['CLASS_LABEL']

    # Split the dataset into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the testing data
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Random Forest Classifier (Assuming best model from Task 2)
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=False,
        random_state=42
    )
    rf_clf.fit(X_train_scaled, y_train)

    # Evaluate the model (Optional)
    from sklearn.metrics import classification_report, accuracy_score

    y_pred = rf_clf.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Initialize the SHAP explainer
    explainer = shap.TreeExplainer(rf_clf)

    # Compute SHAP values for the test set
    shap_values = explainer.shap_values(X_test_scaled)

    # Convert scaled features back to DataFrame for better readability in SHAP
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Summary plot for class 1 (Phishing)
    shap.summary_plot(shap_values[:,:,1], X_test_df, plot_type="bar") # shap_values[1][:,:,1] selects the SHAP values for class 1

    # Dependence plot for a specific feature, e.g., 'PctExtHyperlinks'
    shap.dependence_plot('PctExtHyperlinks', shap_values[:,:,1], X_test_df)
    # Select an instance to explain
    instance = X_test_scaled[0]
    instance_df = X_test_df.iloc[0]

    # Generate SHAP values for the instance
    shap.force_plot(explainer.expected_value[1], shap_values[0,:,1], instance_df, matplotlib=True)   

    # Assuming 'rf_clf' is your trained Random Forest classifier from Task 2
    model_filename = 'phishing_detector.pkl'
    joblib.dump(rf_clf, model_filename)
    print(f"Model saved as {model_filename}")

# Run the Flask application  
if __name__ == '__main__':   
    task_1_2()  
    task_3()
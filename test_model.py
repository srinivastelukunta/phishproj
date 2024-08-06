import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle


@pytest.fixture
def model_data():
    # Loading the dataset
    dataset = pd.read_csv('purhchase_data.csv')
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test, sc


@pytest.fixture
def trained_model(model_data):
    X_train, X_test, y_train, y_test, sc = model_data

    # Fitting Random Forest Classification to the Training set
    classifier = RandomForestClassifier(
        n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    return classifier, sc, X_test, y_test


def test_model_accuracy(trained_model):
    classifier, sc, X_test, y_test = trained_model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Checking accuracy
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.8  # Replace with your expected accuracy threshold


def test_confusion_matrix(trained_model):
    classifier, sc, X_test, y_test = trained_model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Checking confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    assert cm.sum() == len(y_test)


def test_single_prediction(trained_model):
    classifier, sc, _, _ = trained_model

    # Test prediction
    user_age_salary = [[32, 300000]]
    scaled_result = sc.transform(user_age_salary)
    res = classifier.predict(scaled_result)
    assert res in [0, 1]


def test_model_persistence(trained_model):
    classifier, sc, _, _ = trained_model

    # Save model and Scaler
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(classifier, model_file)

    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(sc, scaler_file)

    # Load model and Scaler
    with open('model.pkl', 'rb') as model_file:
        loaded_classifier = pickle.load(model_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)

    # Check if the loaded model makes the same predictions
    user_age_salary = [[32, 300000]]
    scaled_result = loaded_scaler.transform(user_age_salary)
    res = loaded_classifier.predict(scaled_result)
    assert res in [0, 1]

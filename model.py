import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
df = pd.read_csv('Phishing_Legitimate_full.csv')

# Preprocess the dataset
df.drop_duplicates(inplace=True)
label_encoder = LabelEncoder()
df['CLASS_LABEL'] = label_encoder.fit_transform(df['CLASS_LABEL'])

# Separate features and target
X = df.drop('CLASS_LABEL', axis=1)
y = df['CLASS_LABEL']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and fit the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train a simple model (e.g., Logistic Regression)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler to .pkl files
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler have been saved as model.pkl and scaler.pkl.")

import numpy as np
import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from LogisticRegression import LogisticRegression
from ANN import ANN


def train_model():
    # function to calculate accuracy
    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test)

    # Read data
    train = pd.read_csv('train.csv')

    # Data preprocessing
    train.bmi = train.bmi.fillna(round(train.bmi.mean(), 2), axis=0)
    train_df = train[train.columns[1:-1]]

    cat_df = train_df[['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']]
    cat_df = cat_df.astype('category')
    cat_df = cat_df.apply(lambda x: x.cat.codes)

    train_df[cat_df.columns] = cat_df.copy()

    # # Resampling
    X = train_df.values
    y = train.stroke.values

    # Splitting Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling the values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # LogReg Model
    clf1 = LogisticRegression()
    clf1.fit(X_train, y_train)

    # ANN Model
    clf2 = ANN()
    clf2.fit(X_train, y_train)

    return clf1, clf2


# Define the make_prediction function
def make_prediction(user_data):
    # Ensemble: Soft Voting
    clf1, clf2 = train_model()
    pred1 = clf1.predictProb(user_data)[0]
    pred2 = clf2.make_predictions_probability(user_data)[0][0]
    pred = (pred1 + pred2) / 2
    if pred > 0.5:
        prediction = 1
    else:
        prediction = 0

    # Interpret the prediction (e.g., 0: low risk, 1: high risk)
    if prediction == 0:
        risk_label = "Low"
    else:
        risk_label = "High"

    return risk_label, pred


def collect_and_predict():
    try:
        # Collect inputs from entries
        inputs = [float(entry.get()) for entry in entries]
        # Convert inputs to numpy array and reshape for the model
        inputs_array = [inputs, [0, 66, 0, 0, 0, 2, 0, 97.51, 21.5, 1], [1, 66, 1, 1, 1, 2, 1, 100, 21.3, 1], [0, 66, 0, 0, 0, 2, 0, 97.51, 21.5, 1]]
        # Standardize the input data (assuming you used StandardScaler during training)
        scaler = StandardScaler()
        inputs_array_scaled = scaler.fit_transform(inputs_array)
        # Predict using the custom make_prediction function
        risk_label, probability = make_prediction(inputs_array_scaled)
        # Show the prediction result
        messagebox.showinfo("Prediction Result", f"Risk Level: {risk_label}\nProbability: {probability}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers.")


def create_gui():
    root = tk.Tk()
    root.title("Stroke Risk Predictor")

    global entries
    entries = []

    labels = [
        'gender (1 for Male, 0 for Female)',
        'age',
        'hypertension (0 for No, 1 for Yes)',
        'heart_disease (0 for No, 1 for Yes)',
        'ever_married (0 for No, 1 for Yes)',
        'work_type (0 for Govt, 1 for Never-worked, 2 for Private, 3 for Self-employed, 4 for Children)',
        'Residence_type (0 for Rural, 1 for Urban)',
        'avg_glucose_level',
        'bmi',
        'smoking_status (0 for Unknown, 1 for Formerly Smoked, 2 for Never Smoked, 3 for Smoke)'
    ]
    max_label_length = max(len(label) for label in labels)

    for i, label_text in enumerate(labels):
        frame = tk.Frame(root, padx=10, pady=10)
        frame.pack(anchor='w', fill='x')
        label = tk.Label(frame, text=f"{label_text}:", width=max_label_length, anchor='w')
        label.pack(side=tk.LEFT, padx=5)
        entry = tk.Entry(frame)
        entry.pack(side=tk.LEFT, fill='x', expand=True, padx=5)
        entries.append(entry)

    submit_button = tk.Button(root, text="Submit", command=collect_and_predict)
    submit_button.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    create_gui()

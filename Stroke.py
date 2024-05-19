import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from LogisticRegression import LogisticRegression
from ANN import ANN


# function to calculate accuracy
def accuracy(y_test, y_pred):
    return np.sum(y_test==y_pred) / len(y_test)

# Read data
train = pd.read_csv('train.csv')

# Data preprocessing
train.bmi = train.bmi.fillna(round(train.bmi.mean(),2), axis=0)
train_df = train[train.columns[1:-1]]

cat_df = train_df[['gender', 'ever_married','work_type', 'Residence_type','smoking_status']]
cat_df = cat_df.astype('category')
cat_df = cat_df.apply(lambda x : x.cat.codes)

train_df[cat_df.columns] = cat_df.copy()

# Resampling
X = train_df.values
y = train.stroke.values

rus = RandomOverSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X,y)

class_counts = {i : len(y_resampled[y_resampled==i]) for i in np.unique(y_resampled)}
print(f'Instances of the class after re-sampling : {tuple(class_counts.items())}')

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Scaling the values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# LogReg Model
clf1 = LogisticRegression()
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
print("Model Prediction using Logistic Regression: ")
print("Prediction: ",y_pred)
y_pred_prob = clf1.predictProb(X_test)
print("Probability: ", y_pred_prob)
acc1 = accuracy(y_test, y_pred)
print("Accuracy: ", acc1)

print()

# ANN Model
clf2 = ANN()
clf2.fit(X_train, y_train)
y_pred1_prob, y_pred1 = clf2.make_predictions(X_test)
print("Model Prediction using ANN: ")
print("Prediction: ",y_pred1)
print("Probability: ", y_pred1_prob)
acc2 = accuracy(y_test, y_pred1)
print("Accuracy: ", acc2)

print()

# print(confusion_matrix(y_test, y_pred))

# arr = pd.DataFrame({
#      'gender': [0],
#      'age': [66],
#      'hypertension': [0],
#      'heart_disease': [0],
#      'ever_married': [0],
#      'work_type	': [2],
#      'Residence_type': [0],
#      'avg_glucose_level': [97.51],
#      'bmi': [21.3],
#      'smoking_status': [1],
#     })

arr = [[1, 90, 1, 1, 1, 2, 1, 100.51, 40.5, 2], [0, 66, 0, 0, 0, 2, 0, 97.51, 21.5, 1], [1, 66, 1, 1, 1, 2, 1, 100, 21.3, 1], [0, 66, 0, 0, 0, 2, 0, 97.51, 21.5, 1]]
arr = scaler.fit_transform(arr)

print("Using Logistic Regression: ")
probaLR = clf1.predictProb(arr)[0]
predLR = clf1.predict(arr)[0]
print("Probability: ", probaLR)
print("Prediction: ", predLR)

print()

print("Using ANN: ")
probaANN, predANN = clf2.make_predictions(arr)
probaANN = probaANN[0][0]
predANN = predANN[0]
print("Probability: ", probaANN)
print("Prediction: ", predANN)


# Ensemble Method (Soft Voting)
def soft_voting(proba1, proba2):
    result = (proba1 + proba2) / 2
    if result >= 0.5:
        return 1
    else:
        return 0


print()
print("Overall Probability: ", (probaLR + probaANN) / 2)
print("Overall Prediction: ", soft_voting(probaLR, probaANN))

# Error Percentage
print(confusion_matrix(y_test, y_pred))
print(confusion_matrix(y_test, y_pred1))
misclassified_count = len(y_test[y_test != y_pred])
total_cases = len(y_test)
error_rate = misclassified_count / total_cases * 100
print(f"{misclassified_count} misclassified cases out of {total_cases}, error rate : {round(error_rate,2)}%")
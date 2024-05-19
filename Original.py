import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

# Read data
train = pd.read_csv('train.csv')
#print(train.head())

# Data preprocessing
train.bmi = train.bmi.fillna(round(train.bmi.mean(),2), axis=0)
train_df = train[train.columns[1:-1]]

cat_df = train_df[['gender', 'ever_married','work_type', 'Residence_type','smoking_status']]
cat_df = cat_df.astype('category')
cat_df = cat_df.apply(lambda x : x.cat.codes)
print(cat_df.head())

train_df[cat_df.columns] = cat_df.copy()
train_df.head()

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
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# function to calculate accuracy
def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


acc = accuracy(y_test, y_pred)
print(acc)

arr = pd.DataFrame({
    'gender': [0],
    'age': [42],
    'hypertension': [0],
    'heart_disease': [0],
    'ever_married': [1],
    'work_type	': [2],
    'Residence_type': [0],
    'avg_glucose_level': [103],
    'bmi': [40.3],
    'smoking_status': [0],
})
# 0	42.0	0	0	1	2	0	103.00	40.3	0
prediction = clf.predict(arr)
print(prediction)
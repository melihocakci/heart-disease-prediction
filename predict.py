# %%
import pandas as pd # for data manipulation
import numpy as np
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

import matplotlib.pyplot as plt

# Differnt types of Naive Bayes Classifiers
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# %%
df = pd.read_csv('heart_2020_cleaned.csv')

df['BMI'] = pd.qcut(df['BMI'], 5, duplicates='drop')
df['PhysicalHealth'] = pd.qcut(df['PhysicalHealth'], 5, duplicates='drop')
df['MentalHealth'] = pd.qcut(df['MentalHealth'], 5, duplicates='drop')
df['SleepTime'] = pd.qcut(df['SleepTime'], 5, duplicates='drop')

y = df['HeartDisease']
x = df.drop('HeartDisease', axis=1)

enc = LabelEncoder()
y = enc.fit_transform(y)

enc = OrdinalEncoder()
x = enc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

# %%
model = DecisionTreeClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

report = classification_report(y_test, y_pred, target_names=['No','Yes'], output_dict=True)

print("Model: ", model)
print('Classification Report:\n\n', classification_report(y_test, y_pred, target_names=['No','Yes']))

# %%
model = CategoricalNB()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

report = classification_report(y_test, y_pred, target_names=['No','Yes'], output_dict=True)

print("Model: ", model)
print('Classification Report:\n\n', classification_report(y_test, y_pred, target_names=['No','Yes']))

mat = confusion_matrix(y_test, y_pred, normalize='true')

tmp = mat[0][0]
mat[0][0] = mat[1][1]
mat[1][1] = tmp

tmp = mat[0][1]
mat[0][1] = mat[1][0]
mat[1][0] = tmp

plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(mat, annot=True)

plt.show()

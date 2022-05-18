import pandas as pd # for data manipulation
import numpy as np
from sklearn import naive_bayes

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report, confusion_matrix # for model evaluation metrics
from sklearn.preprocessing import OrdinalEncoder # for encoding categorical features from strings to number arrays
from sklearn.preprocessing import LabelEncoder

import plotly.express as px  # for data visualization
import plotly.graph_objects as go # for data visualization

import matplotlib.pyplot as plt

# Differnt types of Naive Bayes Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

################################################################################################################

df = pd.read_csv('heart_2020_cleaned.csv')

df['BMI'] = pd.qcut(df['BMI'], 6, duplicates='drop')
df['PhysicalHealth'] = pd.qcut(df['PhysicalHealth'], 6, duplicates='drop')
df['MentalHealth'] = pd.qcut(df['MentalHealth'], 6, duplicates='drop')
df['SleepTime'] = pd.qcut(df['SleepTime'], 6, duplicates='drop')

y = df['HeartDisease']
x = df.drop('HeartDisease', axis=1)

enc = LabelEncoder()
y = enc.fit_transform(y)

enc = OrdinalEncoder()
x = enc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

################################################################################################################

model = naive_bayes(max_depth=890, max_features='log2', min_samples_leaf=6)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

################################################################################################################

print("Model: ", model)
print('--------------------------------------------------------')
report = classification_report(y_test, y_pred, target_names=['No','Yes'])
print('Classification Report:\n\n', report)
print('--------------------------------------------------------')
print('Confusion Matrix:\n\n', confusion_matrix(y_test, y_pred))

################################################################################################################

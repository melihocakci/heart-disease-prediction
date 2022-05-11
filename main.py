from random import gauss
import pandas as pd # for data manipulation
import numpy as np

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report, confusion_matrix # for model evaluation metrics
from sklearn.preprocessing import OrdinalEncoder # for encoding categorical features from strings to number arrays
from sklearn.preprocessing import LabelEncoder

import plotly.express as px  # for data visualization
import plotly.graph_objects as go # for data visualization

# Differnt types of Naive Bayes Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('heart_2020_cleaned.csv')
# i = 0
# for column in df:
#     print(i)
#     i += 1
#     print(df[column].value_counts())
#     print('-------------------------------------------------')

df['BMI'] = pd.qcut(df['BMI'], 5, duplicates='drop')
df['PhysicalHealth'] = pd.qcut(df['PhysicalHealth'], 5, duplicates='drop')
df['MentalHealth'] = pd.qcut(df['MentalHealth'], 5, duplicates='drop')
df['SleepTime'] = pd.qcut(df['SleepTime'], 5, duplicates='drop')

df = df.to_numpy()

gaussian = df[:,[1,5,6,14]]
bernoulli = df[:,[2,3,4,7,8,12,15,16,17]]
categorical = df[:,[2,3,4,7,8,9,10,11,12,13,15,16,17]]
all = df[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]

y = df[:,0]

############################
x =  all

model = CategoricalNB()
############################

enc = LabelEncoder()
y = enc.fit_transform(y)

enc = OrdinalEncoder()
x = enc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(model)
print('--------------------------------------------------------')
print('Classification Report:\n\n', classification_report(y_test, y_pred, target_names=['No','Yes']))
print('--------------------------------------------------------')
print('Confusion Matrix:\n\n', confusion_matrix(y_test, y_pred))
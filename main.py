from ast import Or
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.preprocessing import OrdinalEncoder # for encoding categorical features from strings to number arrays
from sklearn.preprocessing import LabelEncoder

# Differnt types of Naive Bayes Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB

df = pd.read_csv('heart_2020_cleaned.csv')
df = df.to_numpy()

#x = df[:,[1,5,6,14]] # Gaussian
#x = df[:,[2,3,4,7,8,12,15,16,17]] # Bernoulli
x = df[:,[9,10,11,13]] # Categorical
y = df[:,0]

enc = LabelEncoder()
y = enc.fit_transform(y)

enc = OrdinalEncoder()
x = enc.fit_transform(x)

a = x[0,0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#model = GaussianNB()
#model = BernoulliNB()
model = CategoricalNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))

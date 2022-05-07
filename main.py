import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('heart_2020_cleaned.csv')
data = df[['BMI','PhysicalHealth','MentalHealth','SleepTime']]
target = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=0)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

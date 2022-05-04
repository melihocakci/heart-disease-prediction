import pandas as pd

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('heart_2020_cleaned.csv')
data = df[['BMI','PhysicalHealth','MentalHealth','SleepTime']]
target = df['HeartDisease']

model = GaussianNB()
model.fit(data, target)

print(model)

expected = target
predicted = model.predict(data)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

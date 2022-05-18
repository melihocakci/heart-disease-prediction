import pandas as pd # for data manipulation
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

################################################################################################################

df = pd.read_csv('heart_2020_cleaned.csv')

################################################################################################################

print(df)

################################################################################################################

plt.style.use('fivethirtyeight')

num = df['HeartDisease'].value_counts()

plt.bar(['Yes', 'No'], [num.Yes, num.No])

plt.xlabel('Has Heart Disease')
plt.ylabel('Number of People')
plt.title('Number of People With Heart Disease')

plt.tight_layout()

plt.savefig('./fig/HeartDisease.png')

plt.clf()

################################################################################################################


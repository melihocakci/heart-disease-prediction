# %%
import pandas as pd # for data manipulation
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

# %%
df = pd.read_csv('heart_2020_cleaned.csv')

# %%
print(df.dtypes)

# %%
print(df['HeartDisease'].value_counts())

# %%
df.describe()

# %%
print(df.info())

# %%
truedf = df.query("HeartDisease == 'Yes'")
falsedf = df.query("HeartDisease == 'No'")

# %%
plt.style.use('ggplot')

num = df['HeartDisease'].value_counts()

plt.bar(['Yes', 'No'], [truedf.shape[0], falsedf.shape[0]])

plt.xlabel('Has Heart Disease')
plt.ylabel('Number of People')
plt.title('Number of People With Heart Disease')

plt.show()

plt.clf()



# %%

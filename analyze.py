# %%
import pandas as pd # for data manipulation
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

# %%
df = pd.read_csv('heart_2020_cleaned.csv')

# %%
print(df.info())

# %%
plt.clf()

plt.style.use('fivethirtyeight')

heartdis = len(df.query("HeartDisease == 'Yes'").index)
noheartdis = len(df.query("HeartDisease == 'No'").index)

plt.figure(dpi=100)

#plt.bar()
plt.bar('non-diseased', noheartdis)
plt.bar('diseased', heartdis)
#plt.bar(['has heart disease', 'no heart disease'], [heartdis, noheartdis])

plt.ylabel('quantity')

plt.savefig('fig/heartdis.png', bbox_inches='tight')
plt.show()

plt.clf()

# %%
plt.style.use('fivethirtyeight')

smoker = df.query("Smoking == 'Yes'")
nonsmoker = df.query("Smoking == 'No'")

smokerratio = len(smoker.query("HeartDisease == 'Yes'").index) / len(smoker.index)
nonsmokerratio = len(nonsmoker.query("HeartDisease == 'Yes'").index) / len(nonsmoker.index)

plt.figure(dpi=100)

plt.bar('non-smoker', nonsmokerratio)
plt.bar('smoker', smokerratio)
#plt.bar(['smoker', 'non-smoker'], [smokerratio, nonsmokerratio])

plt.ylabel('heart disease rate')

plt.savefig('fig/smoker.png', bbox_inches='tight')
plt.show()

plt.clf()

# %%
plt.style.use('fivethirtyeight')

drinker = df.query("AlcoholDrinking == 'Yes'")
nondrinker = df.query("AlcoholDrinking == 'No'")

drinkerratio = len(drinker.query("HeartDisease == 'Yes'").index) / len(drinker.index)
nondrinkerratio = len(nondrinker.query("HeartDisease == 'Yes'").index) / len(nondrinker.index)

plt.figure(dpi=100)

plt.bar('non-drinker', nondrinkerratio)
plt.bar('drinker', drinkerratio)
#plt.bar(['drinks alcohol', 'does not drink alcohol'], [drinkerratio, nondrinkerratio])

plt.ylabel('heart disease rate')

plt.savefig('fig/drinker.png', bbox_inches='tight')
plt.show()

plt.clf()



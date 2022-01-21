import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split, cross_val_predict, cross_val_score

# Read dataset from csv file
my_data = pd.read_csv(
    'F:/Coding/Python/Diabates-tensorflow/backend/dataset/diabetes.csv')
my_frame = pd.DataFrame(my_data)
my_frame.head()
my_frame.shape
my_frame.info()
my_frame.describe()
my_frame.isnull().sum()
my_frame.dtypes
my_frame.columns.unique()
fig = plt.figure(figsize=(12, 8))
sns.heatmap(my_frame.isnull())

fig = plt.figure(figsize=(12, 8))
plt.hist(my_frame['Pregnancies'], bins=5, color='green')
fig.suptitle('Pregnancies', fontsize=26)
plt.show()

fig = plt.figure(figsize=(12, 8))
plt.hist(my_frame['Glucose'], bins=5, color='green')
fig.suptitle('Glucose', fontsize=26)
plt.show()

fig = plt.figure(figsize=(12, 8))
plt.hist(my_frame['BloodPressure'], bins=5, color='green')
fig.suptitle('BloodPressure', fontsize=26)
plt.show()

fig = plt.figure(figsize=(12, 8))
plt.hist(my_frame['Age'], bins=5, color='green')
fig.suptitle('Age', fontsize=26)
plt.show()

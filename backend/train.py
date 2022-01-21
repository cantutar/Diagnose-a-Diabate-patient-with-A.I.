import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import RobustScaler

# Read dataset from csv file
df = pd.read_csv(
    'F:/Coding/Python/Diabates-tensorflow/backend/dataset/diabetes.csv')
my_frame = pd.DataFrame(df)
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

# The missing values of CSV will be filled with the median values of each variable. (Saw this one on Tensorflow.js docs.)


def median_target(var):

    temp = df[df[var].notnull()]

    temp = temp[[var, 'Outcome']].groupby(
        ['Outcome'])[[var]].median().reset_index()

    return temp


columns = df.columns
columns = columns.drop("Outcome")
print(columns)

median_target('Glucose')

for feature in df:

    Q1 = df[feature].quantile(0.05)
    Q3 = df[feature].quantile(0.95)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    if df[(df[feature] > upper)].any(axis=None):
        print(feature, "yes")
    else:
        print(feature, "no")

# FEATURES OF DATA
NewBMI = pd.Series(["Underweight", "Normal", "Overweight",
                   "Obesity 1", "Obesity 2", "Obesity 3"], dtype="category")

df["NewBMI"] = NewBMI

df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]

df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
df.loc[df["BMI"] > 39.9, "NewBMI"] = NewBMI[5]
df.head()


def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"


df.head()

df["NewInsulinScore"] = df.apply(set_insulin, axis=1)
df.head()

NewGlucose = pd.Series(["Low", "Normal", "Overweight",
                       "Secret", "High"], dtype="category")

df["NewGlucose"] = NewGlucose

df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]

df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99),
       "NewGlucose"] = NewGlucose[1]

df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126),
       "NewGlucose"] = NewGlucose[2]

df.loc[df["Glucose"] > 126, "NewGlucose"] = NewGlucose[3]
df.head()

df = pd.get_dummies(
    df, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)
df.head()
categorical_df = df[['NewBMI_Obesity 1', 'NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight', 'NewBMI_Underweight',
                     'NewInsulinScore_Normal', 'NewGlucose_Low', 'NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret']]

# Feature Standartization
y = df["Outcome"]
X = df.drop(["Outcome", 'NewBMI_Obesity 1', 'NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight', 'NewBMI_Underweight',
             'NewInsulinScore_Normal', 'NewGlucose_Low', 'NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret'], axis=1)
cols = X.columns
index = X.index
y.head()
X.head()
print(cols, index)
transformer = RobustScaler().fit(X)
X = transformer.transform(X)
X = pd.DataFrame(X, columns=cols, index=index)
X.head()
X = pd.concat([X, categorical_df], axis=1)
X.head()

# algorithm usage and results
models = []
models.append(('LR', LogisticRegression(random_state=42)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=42)))
models.append(('RF', RandomForestClassifier(random_state=42)))
models.append(('SVM', SVC(gamma='auto', random_state=42)))
models.append(('XGB', GradientBoostingClassifier(random_state=42)))
models.append(("LightGBM", LGBMClassifier(random_state=42)))

# evaluate each model in turn
results = []
names = []

for name, model in models:
    # K-10 fold algorithm check
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    cv_results = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

print("exiting...")

import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
df = pd.read_csv("/Users/elvis/Downloads/creditcard.csv")
df.head()
df.info
df.isnull().sum()
from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
df["Amount"] = Sc.fit_transform(pd.DataFrame(df["Amount"]))
df.head()
df=df.drop(["Time"], axis = 1)
df.head()
df.duplicated().any()
df.shape()
df = df.drop_duplicates()
df.shape()

fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(fraud)))
print('Valid Transactions: {}'.format(len(valid)))
X = df.drop('Class',axis=1)
y = df['Class']
y.value_counts()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3 , random_state = 42)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
acc = accuracy_score(y_test , y_pred)
print("The accuracy is {}".format(acc))

prec = precision_score(y_test , y_pred)
print("The precision is {}".format(prec))

rec = recall_score(y_test , y_pred)
print("The recall is {}".format(rec))

f1 = f1_score(y_test , y_pred)
print("The F1-Score is {}".format(f1))
import joblib
joblib.dump(rf , "Credit_card_Fraud_Model")
model = joblib.load("Credit_card_Fraud_Model")
prediction  = model.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
prediction
if prediction==0:
  print("Valid Transaction")
else:
  print("Fraud Transaction")

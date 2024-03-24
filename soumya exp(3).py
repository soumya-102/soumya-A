import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
df = pd.read_csv("/Users/elvis/Downloads/IRIS.csv")
df.head()
print("First few rows of the dataset:")
print(df.head())

print("\nSummary statistics of the dataset:")
print(df.describe())

print("\nDistribution of the species variable:")
print(df['species'].value_counts())

plt.figure(figsize=(8, 6))
plt.hist(df['species'], alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Species')
plt.ylabel('Frequency')
plt.title('Distribution of Species Variable')
plt.grid(True)
plt.show()
plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[:-1]): 
    plt.subplot(2, 2, i + 1)
    plt.hist(df[feature], bins=20, color='lightblue', edgecolor='black')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()
with warnings.catch_warnings():
warnings.simplefilter("ignore", category=FutureWarning)
sns.pairplot(df, hue='species', diag_kind="hist", corner=True, palette='hls')
sns.pairplot(df , hue='species' , diag_kind="hist" , corner=True , palette = 'hls');
sns.pairplot(df, hue='species')
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, orient='h')
plt.title('Boxplot of Iris Dataset Features')
plt.show()
numeric_cols = df.drop('species', axis=1)
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numeric Features in Iris Dataset')
plt.show()

plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data=df, x=feature, hue='species', kde=True, bins=20)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.violinplot(data=df, x='species', y=feature)
    plt.xlabel('Species')
    plt.ylabel(feature)
plt.tight_layout()
plt.show()
X = df.drop('species', axis=1)
y = df['species']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of training set:", X_train.shape, y_train.shape)
print("Shape of testing set:", X_test.shape, y_test.shape)
side heading- selection of ML model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression model:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

logistic_model = LogisticRegression(max_iter=1000)
tree_model = DecisionTreeClassifier()
forest_model = RandomForestClassifier()
svm_model = SVC()
knn_model = KNeighborsClassifier()
models = [logistic_model, tree_model, forest_model, svm_model, knn_model]
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'k-NN']
for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {name}: {accuracy}")
    best_model_idx = np.argmax([accuracy_score(y_test, model.predict(X_test)) for model in models])
best_model_name = model_names[best_model_idx]
print(f"\nThe best model based on accuracy is: {best_model_name}")

models = {'Logistic Regression': LogisticRegression(max_iter=1000),
'Decision Tree': DecisionTreeClassifier(),
'Random Forest': RandomForestClassifier(),
'SVM': SVC(),
'k-NN': KNeighborsClassifier()}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    best_model = max(results, key=results.get)
best_accuracy = results[best_model]
print("Model Comparisons:")
for name, accuracy in results.items():
    print(f"{name}: Accuracy = {accuracy:.4f}")

print("\nThe best model based on accuracy is:", best_model, "with accuracy =", best_accuracy)
features = X_train.columns

importances = forest_model.feature_importances_

feat_imp = pd.Series(importances,index=features).sort_values()

feat_imp.tail().plot(kind='barh')
plt.xlabel('Importance Ratio')
plt.ylabel('Attributes')
plt.title('Features Importances');
sideheading - visualize results
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import warnings

label_encoder = LabelEncoder()
y_train_numeric = label_encoder.fit_transform(y_train)
y_test_numeric = label_encoder.transform(y_test)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train_numeric)
new_data = np.array([[5.1, 3.5, 1.4, 0.2],  
                      [6.2, 2.8, 4.8, 1.8],
                      [7.3, 3.1, 6.3, 2.3]])

predicted_classes = model.predict(new_data)
iris_target_names = ['setosa', 'versicolor', 'virginica']
print("Predicted Classes:")
for data, prediction in zip(new_data, predicted_classes):
    print(f"Iris data: {data} -> Predicted Class: {iris_target_names[prediction]}")

y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test_numeric, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(iris_target_names))
plt.xticks(tick_marks, iris_target_names, rotation=45)
plt.yticks(tick_marks, iris_target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
for i in range(len(iris_target_names)):
    for j in range(len(iris_target_names)):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

plt.tight_layout()
plt.show()

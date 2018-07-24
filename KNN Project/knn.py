import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('KNN_Project_Data')

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(data.drop('TARGET CLASS',axis=1))
scaled_feat = scaler.transform(data.drop('TARGET CLASS',axis=1))
df_scaled = pd.DataFrame(scaled_feat,columns=data.columns[:-1])
df_scaled.head()

# Create training and test sets
from sklearn.model_selection import train_test_split

X = df_scaled
y = data['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
print('Data split into training and test sets.\n')
print('Training set size: {}\nTest set size: {}\n'.format(X_train.shape[0],X_test.shape[0]))

# Run KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
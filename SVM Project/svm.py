import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import and explore data
iris = sns.load_dataset('iris')
print("Iris dataset loaded...")
print(iris.head())
wait = input("PRESS ENTER TO CONTINUE.")

print('Exploring data...')
sns.set_style('darkgrid')
sns.pairplot(iris,hue="species")
plt.show()
wait = input("PRESS ENTER TO CONTINUE.")

print("Setosa seems most separable.  Isolated setosa by sepal length/width...")
setosa = iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_length'],setosa['sepal_width'],cmap='plasma',shade=True,shade_lowest=False)
plt.show()
wait = input("PRESS ENTER TO CONTINUE.")

# Train test split
from sklearn.model_selection import train_test_split
X = iris.drop('species',axis=1)
y = iris['species']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
print("Split data into Training and Test sets.")
print("Training Set Size: {}\nTest Set Size: {}\n".format(X_train.shape[0],X_test.shape[0]))
wait = input("PRESS ENTER TO CONTINUE.")

# Perform machine learning regression
print('Running SVC using default parameters.')
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
pred = svc.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
print('Printing metrics...')
print('Confusion Matrix: \n{}\n'.format(confusion_matrix(y_test,pred)))
print('Classification Report:\n{}\n'.format(classification_report(y_test,pred)))
wait = input("PRESS ENTER TO CONTINUE.")

# Optional: Use Grid search to improve model

from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[0.0001,0.001,0.01,0.1,1,10]}
grid_cv = GridSearchCV(SVC(),param_grid,verbose=2,refit=True)
grid_cv.fit(X_train,y_train)
grid_pred = grid_cv.predict(X_test)

print('New metrics:\n')
print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))
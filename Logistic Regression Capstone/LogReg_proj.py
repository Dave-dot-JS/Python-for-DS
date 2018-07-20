import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import data
data = 'advertising.csv'
ad_data = pd.read_csv(data)
print('The data set {} has been imported.'.format(data))

# Create training and test sets and run algorithm
X = ad_data.drop(['Clicked on Ad','Ad Topic Line','City','Country','Timestamp'], axis=1)
y = ad_data['Clicked on Ad']

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print('Data split into training and test sets.\n')
print('Training set size: {}\nTest set size: {}\n'.format(X_train.shape[0],X_test.shape[0]))


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train, y_train)
print('Running Logistic Regression...')

y_pred = logreg.predict(X_test)

from sklearn.metrics import classification_report
print('Regression run successfully.')
print('Classification Report:\n')
print(classification_report(y_test, y_pred))

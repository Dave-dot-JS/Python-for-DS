# First let's clean the training set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')
print('Data is loaded...')

# Some missing Ages, impute missing values with average by Passenger Class
# df used to generalize, set DF variable to train or test set before running
def impute_age(cols):
	Age = cols[0]
	Pclass = cols[1]

	if pd.isnull(Age):
		if Pclass == 1:
			return df[df['Pclass'] == 1]['Age'].mean()
		elif Pclass == 2:
			return df[df['Pclass'] == 2]['Age'].mean()
		else:
			return df[df['Pclass'] == 3]['Age'].mean()
	else:
		return Age

df = train
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

df = test
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)


# Cabin missing too many values, drop the column as well as any remaining NaNs
train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)

test.drop('Cabin', axis=1, inplace=True)
test.dropna(inplace=True)

# Replace classification variables with dummy variables
sex_train = pd.get_dummies(train['Sex'],drop_first=True)
embark_train = pd.get_dummies(train['Embarked'],drop_first=True)

sex_test = pd.get_dummies(test['Sex'],drop_first=True)
embark_test = pd.get_dummies(test['Embarked'],drop_first=True)


# Concatenate new columns and drop unused columns
pd.concat([train,sex_train,embark_train],axis=1)
train.drop(['Sex','Name','Ticket','Embarked','PassengerId'],axis=1,inplace=True)

pd.concat([test,sex_test,embark_test],axis=1)
test.drop(['Sex','Name','Ticket','Embarked','PassengerId'],axis=1,inplace=True)

print('Data cleaned.  Beginning regression.')

# Set variables for logistic regression and perform regression
X_train = train.drop('Survived',axis=1)
y_train = train['Survived']
X_test = test


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
y_pred = logmodel.predict(X_test)
score = logmodel.score(X_train, y_train)
print('The model has run with an accuracy score of ' + str(score))



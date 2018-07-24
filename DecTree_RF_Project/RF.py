import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import and inspect data
loans = pd.read_csv('loan_data.csv')
print(loans.head())
wait = input("PRESS ENTER TO CONTINUE.")
print(loans.info())
wait = input("PRESS ENTER TO CONTINUE.")
print(loans.describe())
wait = input("PRESS ENTER TO CONTINUE.")

# Plot FICO by credit policy and payment status
plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

loans[loans['credit.policy']==1]['fico'].hist(bins=30,alpha=0.5, color='blue',label='Credit Policy =1')
loans[loans['credit.policy']==0]['fico'].hist(bins=30,alpha=0.5, color='red',label='Credit Policy = 0')

plt.legend()
plt.xlabel('FICO')
plt.show()
wait = input("PRESS ENTER TO CONTINUE.")

plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

loans[loans['not.fully.paid']==1]['fico'].hist(bins=30,alpha=0.5, color='blue',label='Not Fully Paid = 1')
loans[loans['not.fully.paid']==0]['fico'].hist(bins=30,alpha=0.5, color='red',label='Not Fully Paid = 0')

plt.legend()
plt.xlabel('FICO')
plt.show()
wait = input("PRESS ENTER TO CONTINUE.")

plt.figure(figsize=(10,7))
sns.countplot(x='purpose',data=loans,hue='not.fully.paid')
plt.tight_layout()
plt.show()
wait = input("PRESS ENTER TO CONTINUE.")

sns.lmplot(x='fico',y='int.rate',data=loans,col='not.fully.paid',hue='credit.policy')
plt.show()
wait = input("PRESS ENTER TO CONTINUE.")


# Remove categorical features and replace with dummies
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)

# Split test data
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
print("Split data into Training and Test sets.")
print("Training Set Size: {}\nTest Set Size: {}\n".format(X_train.shape[0],X_test.shape[0]))

# Start with decision tree
wait = input('Loading Decision Tree...Press Enter to continue.')
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print('Metrics for Single Decision Tree:\n')
print(classification_report(y_test,pred))
wait = input("PRESS ENTER TO CONTINUE")
print(confusion_matrix(y_test,pred))
wait = input("PRESS ENTER TO CONTINUE")

# Now implement Random Forest model instead
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

pred_rfc = rfc.predict(X_test)
print('Metrics for Random Forest:\n')
print(classification_report(y_test,pred_rfc))
wait = input("PRESS ENTER TO CONTINUE")
print(confusion_matrix(y_test,pred_rfc))
wait = input("PRESS ENTER TO CONTINUE")
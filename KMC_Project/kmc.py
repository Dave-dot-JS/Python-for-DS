import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('College_Data',index_col=0)
data['Grad.Rate']['Cazenovia College'] = 100 # Grad rate was > 100%, corrected
print('Importing data and exploring')
print(data.head())
wait = input('ENTER TO CONTINUE.')
print(data.info())
wait = input('ENTER TO CONTINUE.')
print(data.describe())
wait = input('ENTER TO CONTINUE.')

print('Visualizing Data...')
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data,hue='Private',fit_reg=False,size=6,aspect=1,palette='coolwarm')
plt.show()
wait = input('ENTER TO CONTINUE.')

sns.lmplot(x='Outstate',y='F.Undergrad',data=data,hue='Private',palette='coolwarm',fit_reg=False)
plt.show()
wait = input('ENTER TO CONTINUE.')

from sklearn.cluster import KMeans
kmc = KMeans(n_clusters=2)
kmc.fit(data.drop('Private',axis=1))

def binary(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0

data['Cluster'] = data['Private'].apply(binary)

print('Runnning K-Means Algorithm...\n')
from time import sleep
sleep(2)

from sklearn.metrics import classification_report
print('Classification Report:\n{}'.format(classification_report(data['Cluster'],kmc.labels_)))
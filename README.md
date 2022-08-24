# EV-Market-Analysis
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
df1= pd.read_csv("C:\\Datasets\\Cheapestelectriccars-EVDatabase.csv")
df1.head()
df1=df1.fillna('0')
PriceinUK=[]
for item in df1['PriceinUK']:
    PriceinUK+=[int(item.replace('£','').replace(',',''))]
df1['PriceinUK']=PriceinUK
PriceinGermany=[]
for item in df1['PriceinGermany']:
    PriceinGermany+=[int(item.replace('€','').replace(',',''))]
df1['PriceinGermany']=PriceinGermany
FastChargeSpeed=[]
for item in df1['FastChargeSpeed']:
    FastChargeSpeed+=[int(item.replace(' km/h','').replace('-','0'))]
df1['FastChargeSpeed']=FastChargeSpeed
Efficiency=[]
for item in df1['Efficiency']:
    Efficiency+=[int(item.replace(' Wh/km',''))]
df1['Efficiency']=Efficiency
Range=[]
for item in df1['Range']:
    Range+=[int(item.replace(' km',''))]
df1['Range']=Range
TopSpeed=[]
for item in df1['TopSpeed']:
    TopSpeed+=[int(item.replace(' km/h',''))]
df1['TopSpeed']=TopSpeed
Acceleration=[]
for item in df1['Acceleration']:
    Acceleration+=[float(item.replace(' sec',''))]
df1['Acceleration']=Acceleration
Subtitle=[]
for item in df1['Subtitle']:
    Subtitle+=[float(item.replace('Battery Electric Vehicle | ','').replace(' kWh','').replace('      ',''))]
df1['Subtitle']=Subtitle
sns.countplot(x = 'Drive', data = df1)
sns.countplot(x = 'NumberofSeats', data = df1)
plt.figure(figsize=(8,6))
sns.countplot(x = 'NumberofSeats', hue='Drive', data=df1)
sns.relplot(x="KWH", y="Acceleration", height=6,hue="Drive",data=df1)
sns.jointplot(x=df1["KWH"], y=df1["Efficiency"], kind="hex", color="#4CB391")
import category_encoders as ce
train_df=df1
encoder= ce.OrdinalEncoder(cols=['Drive'],return_df=True,
                           mapping=[{'col':'Drive',
'mapping':{'Front Wheel Drive':1,'Rear Wheel Drive':2,'All Wheel Drive':3}}])
df_train = encoder.fit_transform(train_df)
X1=df1.iloc[:,[5,10]].values
X1
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
kmeans=KMeans(n_clusters=3,random_state=42,init='k-means++')
Y_kmeans=kmeans.fit_predict(X1)
plt.scatter(X1[Y_kmeans==0,0],X1[Y_kmeans==0,1],s=100,c='green',label='Cluster-1')
plt.scatter(X1[Y_kmeans==1,0],X1[Y_kmeans==1,1],s=100,c='orange',label='Cluster-2')
plt.scatter(X1[Y_kmeans==2,0],X1[Y_kmeans==2,1],s=100,c='blue',label='Cluster-3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='black',label='Centroids')
plt.title('Electric Vehicles')
plt.xlabel('Efficiency')
plt.ylabel('PriceinUK')
plt.legend()
plt.show()

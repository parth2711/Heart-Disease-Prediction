from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

df=pd.read_csv('heart_disease_uci.csv')
df.drop(['id','dataset'],axis=1,inplace=True)

df.dropna(inplace=True)

df['sex']=df['sex'].str.lower().map({'male':1,'female':0})
df['cp']=df['cp'].str.lower().map({'typical angina':0,'atypical angina':1,'non-anginal':2,'asymptomatic':3,'typical':4,'atypical':5})
df['fbs']=df['fbs'].map({True:1,False:0})
df['restecg']=df['restecg'].str.lower().map({'normal':0,'lv hypertrophy':1,'st-t abnormality':2})
df['exang']=df['exang'].map({True:1,False:0})
df['slope']=df['slope'].str.lower().map({'flat':0,'upsloping':1,'downsloping':2})
df['thal']=df['thal'].str.lower().map({'normal':0,'fixed defect':1,'reversable defect':2})

# plt.figure(figsize=(10,6))
# plt.plot(df['thal'])
# plt.show()

x=df.drop('num',axis=1)
y=(df['num'] > 0).astype(int)


scaler=StandardScaler()

X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

knnmodel=KNeighborsClassifier(n_neighbors=11,weights='distance')
knnmodel.fit(X_train_scaled,y_train)

with open('model.pkl','wb') as file:
    pickle.dump(knnmodel,file)

with open('scaler.pkl','wb') as file2:
    pickle.dump(scaler,file2)

prediction=knnmodel.predict(X_test_scaled)

print(confusion_matrix(y_test,prediction))
print('\n',classification_report(y_test,prediction))

# err=[]
# for i in range(1,40):
#     knn=KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train_scaled,y_train)
#     pred=knn.predict(X_test_scaled)
#     err.append(np.mean(pred!=y_test))

# plt.figure(figsize=(10,6))
# plt.plot(range(1,40),err,color='blue',linestyle='--',marker='o')
# plt.title('Error vs K')
# plt.xlabel('K')
# plt.ylabel('Error')
# plt.show()
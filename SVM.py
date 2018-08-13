from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
#%%-----------------------------DATA PREPROCESSING-----------------------------------------
train_data = pd.read_csv('train_users_final.csv')
test_data = pd.read_csv('test_users_final.csv')

id_test = test_data['id']
test = test_data.drop(['id'], axis=1)

label = train_data['country_destination']
y = label.values
class_le = LabelEncoder()
y = class_le.fit_transform(y)
X = train_data.drop(['id','country_destination'],axis=1)
X = X.values
#Normalize the data
sc = StandardScaler()
sc.fit(X)
X= sc.transform(X)
#split data into train ans test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
clf =SVC(kernel = 'linear',C=1.0,random_state=2018,probability=True)
clf.fit(X_train,y_train)
#%%----------------------------------------------------------------------
#Fit SVM model
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
svm = LinearSVC()
clf = CalibratedClassifierCV(svm)
clf.fit(X_train, y_train)
sc.fit(X_test)
X_test= sc.transform(X_test)
y_pred = clf.predict(X_test)
#%%----------------------------------------------------------------------
#accuracy of SVM
print("\n")
print("Results Using SVM: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print('accuracy:',accuracy_score(y_test, y_pred) * 100)
#Predict on the test data
sc.fit(test)
test_trans= sc.transform(test)

y_pred_t = clf.predict(test_trans)
y_proba_t = clf.predict_proba(test_trans)
ids = []
cts = []
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx]*5
    cts += class_le.inverse_transform(np.argsort(y_proba_t[i])[::-1])[:5].tolist()
sub_svm = pd.DataFrame(np.column_stack((ids,cts)),columns=['id','country'])
sub_svm.to_csv('sub_svm.csv',index=False)
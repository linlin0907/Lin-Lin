import numpy as np
import pandas as pd
import matplotlib as mpl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys
from pydotplus import graph_from_dot_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#%%-----------------------------------------------------------------------
#Libraries to display decision tree
from sklearn.tree import export_graphviz
import webbrowser
#%%-----------------------------------------------------------------------
data = pd.read_csv('train_users_final.csv')
label = data.loc[:,'country_destination']
X_data = data.drop(['country_destination'], axis=1)
X = X_data.drop(['id'],axis=1)

test_data = pd.read_csv('test_users_final.csv')
id_test = test_data['id']
test = test_data.drop(['id'], axis=1)


class_le = LabelEncoder()
# fit and transform the class
y = class_le.fit_transform(label)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
#%%-----------------------------------------------------------------------
# perform training with gini Index.
# creating the classifier object
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=7, min_samples_leaf=50)

# performing training
clf_gini.fit(X_train, y_train)
#%%-----------------------------------------------------------------------

# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=7, min_samples_leaf=50)

# Performing training
clf_entropy.fit(X_train, y_train)

#%%-----------------------------------------------------------------------
# make predictions
# predicton on test using gini
y_pred_gini = clf_gini.predict(X_test)
# predicton on test using entropy
y_pred_entropy = clf_entropy.predict(X_test)
#%%-----------------------------------------------------------------------
# calculate metrics gini model
print("\n")
print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_gini))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print("\n")
print ('-'*80 + '\n')
# calculate metrics entropy model
print("\n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print ('-'*80 + '\n')
#%%-----------------------------------------------------------------------
# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred_gini)
class_names = data.country_destination.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 10}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()
#%%-----------------------------------------------------------------------
# confusion matrix for entropy model
conf_matrix = confusion_matrix(y_test, y_pred_entropy)
class_names = data.country_destination.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 10}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

#Feature importance for decision tree
feature_importance = clf_gini.feature_importances_
feature_importance = 100*(feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0])+.5
plt.subplot(1,2,2)
plt.barh(pos,feature_importance[sorted_idx],align='center')
plt.yticks(pos,data.columns[sorted_idx])
plt.ylim(150,169)
plt.xlabel('Relative Importance')
plt.title('Feature Importance',fontsize=15)
plt.show()
#%%-----------------------------------------------------------------------
# display decision tree
dot_data = export_graphviz(clf_gini, filled=True, rounded=True, class_names=class_names, feature_names=X.iloc[:, :].columns, out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_gini_1.pdf")
webbrowser.open_new(r'decision_tree_gini_1.pdf')

#%%-----------------------------------------------------------------------

dot_data = export_graphviz(clf_entropy, filled=True, rounded=True, class_names=class_names, feature_names=X.iloc[:, :].columns, out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_entropy_1.pdf")
webbrowser.open_new(r'decision_tree_entropy_1.pdf')

# make prediction with gini
y_pred_gini = clf_gini.predict(test)
y_pred_score_gini = clf_gini.predict_proba(test)

ids = []
cts = []
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx]*5
    cts += class_le.inverse_transform(np.argsort(y_pred_score_gini[i])[::-1])[:5].tolist()

sub_dt = pd.DataFrame(np.column_stack((ids,cts)),columns=['id','country'])
sub_dt.to_csv('sub_dt.csv',index=False)
#make prediction with entropy
y_pred_entropy = clf_entropy.predict(test)
y_pred_score_entropy = clf_entropy.predict_proba(test)
ids = []
cts = []
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx]*5
    cts += class_le.inverse_transform(np.argsort(y_pred_score_entropy[i])[::-1])[:5].tolist()

sub_dt_en = pd.DataFrame(np.column_stack((ids,cts)),columns=['id','country'])
sub_dt_en.to_csv('sub_dt_en.csv',index=False)

#%%------------------------------Random Forest-----------------------------------------
X_train=X_train.values[:,:]
X_test =X_test.values[:,:]
clf = RandomForestClassifier(n_estimators=100)

# perform training
clf.fit(X_train, y_train)
# get feature importances
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, X.columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot= 90, fontsize=15)
plt.title('Feature Importance',fontsize=20,fontname="Times New Roman Bold")
plt.xlim(-0.5,19.5)
# show the plot
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------
#select features to perform training with random forest with k columns
# select the training dataset on k-features
newX_train = X_train[:, clf.feature_importances_.argsort()[::-1][:15]]

# select the testing dataset on k-features
newX_test = X_test[:, clf.feature_importances_.argsort()[::-1][:15]]

#%%-----------------------------------------------------------------------
#perform training with random forest with k columns
# specify random forest classifier
clf_k_features = RandomForestClassifier(n_estimators=100)

# train the model
clf_k_features.fit(newX_train, y_train)
#%%-----------------------------------------------------------------------
#accuracy of random forest
#accuracy of all features
y_pred = clf.predict(X_test)
print("\n")
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

#accuracy of K features
y_pred_k_features = clf_k_features.predict(newX_test)
print("\n")
print("Results Using K features: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_k_features))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_k_features) * 100)
# %%-----------------------------------------------------------------------
# confusion matrix for all features
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['country_destination'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
# %%-----------------------------------------------------------------------
# predicton on test using all features
y_pred_rf = clf.predict(test)
y_pred_score1_rf = clf.predict_proba(test)
ids = []
cts = []
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx]*5
    cts += class_le.inverse_transform(np.argsort(y_pred_score1_rf[i])[::-1])[:5].tolist()

sub_dt = pd.DataFrame(np.column_stack((ids,cts)),columns=['id','country'])
sub_dt.to_csv('sub_rf.csv',index=False)

print ('-'*40 + 'End Console' + '-'*40 + '\n')
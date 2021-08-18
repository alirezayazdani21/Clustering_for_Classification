#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


# In[2]:


from sklearn.datasets import make_classification


# In[3]:


from sklearn.datasets import make_classification

w=.97

X, y = make_classification(n_samples=5000,n_classes=2,n_features=15, random_state=123, n_clusters_per_class=1, weights=[w])


# In[4]:


np.unique(y, return_counts=True)


# In[5]:


sn.histplot(y);


# In[6]:


X


# In[7]:


from sklearn.ensemble import RandomForestClassifier


# In[8]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=42, class_weight='balanced').fit(X,y)


# In[9]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


# In[10]:


clf_preds=pd.DataFrame(clf.predict(X))
clf_pred_probs=pd.DataFrame(clf.predict_proba(X)[:,0])
#clf_preds
clf_pred_probs


# In[11]:


confusion_matrix(y,clf_preds)


# In[12]:


accuracy_score(y,clf_preds)


# In[13]:


roc_auc_score(y,clf_preds)


# In[14]:


plot_confusion_matrix(clf, X, y) 
plt.title("Confusion Matrix for Classification")
plt.show()  


# In[15]:


#######################################################


# In[16]:


X_train=pd.DataFrame(data=X)
X_train


# In[17]:


#use this if you wantto add the pca feature
from sklearn.decomposition import PCA
pca = PCA(n_components=1, random_state=123).fit_transform(X_train)
pca = pd.DataFrame(pca,columns=['pca'])
#pca
#X_train= pd.concat([X_train,pca],axis=1)


# In[18]:


#Isolation forest 


# In[19]:


from sklearn.ensemble import IsolationForest
rng = np.random.RandomState(42)
ISFOR_clus = IsolationForest(n_estimators=1000,max_samples=1000,bootstrap=True,max_features=5, random_state=rng).fit(X_train)
ISFOR_clus_preds=pd.DataFrame(ISFOR_clus.predict(X_train))
ISFOR_clus_preds=ISFOR_clus_preds.replace(1, 0).replace(-1,1)


# In[20]:


#Spectral clustering


# In[21]:


from sklearn.cluster import SpectralClustering
rng = np.random.RandomState(42)
SP_clus = SpectralClustering(n_clusters=2, n_init=1000, random_state=rng)
SP_clus_preds=pd.DataFrame(SP_clus.fit_predict(X_train))


# In[22]:


#One class Svm


# In[23]:


rng = np.random.RandomState(42)
from sklearn.svm import OneClassSVM
SVM_clus = OneClassSVM(gamma='auto').fit(X_train)
#SVM_clus.predict(X_train)

#SVM_clus.score_samples(X_train)
SVM_clus_preds=pd.DataFrame(SVM_clus.fit_predict(X_train))
SVM_clus_preds=SVM_clus_preds.replace(1, 0).replace(-1,1)


# In[24]:


#KMeans


# In[25]:


from sklearn.cluster import KMeans
KMEANS_clus = KMeans(n_clusters=2, n_init=1000, max_iter=1000,random_state=rng).fit(X_train)
KMEANS_clus.labels_
KMEANS_clus_preds=pd.DataFrame(KMEANS_clus.fit_predict(X_train))
KMEANS_clus_preds


# In[26]:


#LocalOutlierFactor


# In[27]:


from sklearn.neighbors import LocalOutlierFactor
LOCOUT_clus = LocalOutlierFactor(n_neighbors=10).fit(X_train)
LOCOUT_clus_preds=pd.DataFrame(LOCOUT_clus.fit_predict(X_train))
LOCOUT_clus_preds=LOCOUT_clus_preds.replace(1, 0).replace(-1,1)
LOCOUT_clus_preds.value_counts()


# In[28]:


#DBScan


# In[29]:


from sklearn.cluster import DBSCAN
DBSC_clus = DBSCAN(eps=3, min_samples=10).fit(X_train)
DBSC_clus_preds=pd.DataFrame(DBSC_clus.fit_predict(X_train))
DBSC_clus_preds=DBSC_clus_preds.replace(1, 0).replace(-1,1)


# In[34]:


DBSC_clus_preds.value_counts()
LOCOUT_clus_preds.value_counts()


# In[31]:


#clus_preds


# In[38]:


finals_preds= pd.concat([clf_preds,clf_pred_probs,ISFOR_clus_preds,SP_clus_preds,SVM_clus_preds,KMEANS_clus_preds,LOCOUT_clus_preds, DBSC_clus_preds],axis=1)
finals_preds.columns=['clf_class','clf_score', 'ISOFOR','SPECTR','SVM-1C','KMEANS','LOCOUT','DBSCAN']
finals_preds


# In[46]:


finals_preds['ENSEMB']= finals_preds[['ISOFOR','SPECTR','SVM-1C','KMEANS','LOCOUT']].mode(axis=1)
finals_preds


# In[47]:


finals_preds['clf_class']


# In[48]:


finals_preds['ENSEMB']


# In[49]:


confusion_matrix(finals_preds['clf_class'],finals_preds['ENSEMB'])


# In[50]:


def cluster_scores(cls):
    ACC,ROC=[round(accuracy_score(finals_preds['clf_class'],finals_preds[cls]),2) , 
       round(roc_auc_score(finals_preds['clf_class'],finals_preds[cls]),2)]
    return print(cls, ': Acc=',ACC, '&', 'ROC=',ROC)


# In[72]:


print(cluster_scores('ISOFOR'),
      cluster_scores('SPECTR'),
      cluster_scores('LOCOUT'),
      cluster_scores('SVM-1C'),
      cluster_scores('DBSCAN'),
      cluster_scores('KMEANS'),
      cluster_scores('ENSEMB'))


# In[63]:


sn.boxplot(x="ENSEMB",y="clf_score",data=finals_preds);


# In[64]:


#sn.boxplot(x="clf_class",y="clf_score",data=finals_preds);


# In[65]:


#finals_preds.describe()


# In[67]:


cm = confusion_matrix(finals_preds['clf_class'],finals_preds['ENSEMB'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion matrix of the clustering');


# In[ ]:





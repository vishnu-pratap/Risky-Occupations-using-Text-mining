# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:08:24 2016

@author: Vishnu
"""
import re
import pandas as pd
import os
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from nltk.corpus import stopwords


os.chdir("D:\Semester2\TextMining\CA")
train=pd.read_excel("osha-labelled.xlsx")
col_list=['Abstract','Occupation']
train=train[col_list]

def preprocessing(text):
    letters=re.sub("[^a-zA-Z]"," ",text)
    words=letters.lower().split()
    stops=set(stopwords.words("english"))
    meaningful_words=[w for w in words if not w in stops]
    return(" ".join(meaningful_words)) 
    
labelled=[]
for row in train.iterrows():
    index,data=row
    labelled.append(preprocessing(data[0]))
    
label=train["Occupation"]
label=label.tolist()
label=[x.encode('UTF8') for x in label]
labelled=[x.encode('UTF8') for x in labelled]
data_labelled=zip(labelled,label)
data_labelled=pd.DataFrame(data_labelled) 
labelled=[]
for row in data_labelled.iterrows():
    index, data = row
    labelled.append(data.tolist())

def create_tfidf_training_data(docs):
    corpus=[d[0] for d in docs]
    y=[d[1] for d in docs]
    vectorizer=TfidfVectorizer(min_df=1)
    X=vectorizer.fit_transform(corpus)
    return X,y
    
X,y=create_tfidf_training_data(labelled)
#vocab=vectorizer.get_feature_names()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.22,random_state=42)
#print(len(X_train.data))
#print(len(y_train))
#print(len(y_test))


from sklearn.naive_bayes import MultinomialNB 
clf=MultinomialNB().fit(X_train,y_train)  
pred=clf.predict(X_test)
print(clf.score(X_test,y_test))
labels = list(set(y_train))
print(metrics.classification_report(y_test, pred, target_names=list(set(y_test))))  


def train_dtc(X, y):
    """
    Create and train the Decision Tree Classifier.
    """
    dtc = DecisionTreeClassifier()
    dtc.fit(X, y)
    return dtc
dt = train_dtc(X_train,y_train)
predDT = dt.predict(X_test)

# Print the classification rate
print(dt.score(X_test, y_test)) 


def train_knn(X, y, n, weight):
    """
    Create and train the k-nearest neighbor.
    """
    knn = KNeighborsClassifier(n_neighbors=n, weights=weight, metric='cosine', algorithm='brute')
    knn.fit(X, y)
    return knn
    
kn = train_knn(X_train, y_train, 5, 'distance')
predKN = kn.predict(X_test)

# Print the classification rate
print(kn.score(X_test, y_test)) 


def train_lr(X, y):
    """
    Create and train the Naive Baye's Classifier.
    """
    lr = LogisticRegression()
    lr.fit(X, y)
    return lr
    
lr = train_lr(X_train, y_train)
predLR = lr.predict(X_test)  
print(lr.score(X_test, y_test)) 


def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm
    
sv = train_svm(X_train, y_train)
predSVM = sv.predict(X_test)

# Print the classification rate
print(sv.score(X_test, y_test))  


columns = ['DecisionTree', 'NearestNeighbor', 'SVM', 'NaiveBayes', 'Logistic']
ensDF = pd.DataFrame(
    {'DecisionTree': predDT, 'NearestNeighbor': predKN, 'SVM': predSVM, 'NaiveBayes': pred, 'Logistic': predLR},
    columns=columns)
print ensDF
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 22:54:33 2016

@author: Vishnu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:08:24 2016

@author: Vishnu
"""
import re
import unicodedata
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
from wordcloud import WordCloud

os.chdir("D:\Semester2\TextMining\CA")
data=pd.read_excel("risk.xlsx")
data=data[data.Abstract!="InspectionOpen DateSICEstablishment Name"]


col_list=['SN','Title']
data=data[col_list]
risk=[]
for row in data.iterrows():
    index,data=row
    risk.append(data.tolist())
    
for r in risk:
    if r[0]==202614046 or r[0]==14240477 or r[0]==200536050 or r[0]==14372981:
        risk.remove(r)
        
risk_title=[d[1] for d in risk]

def preprocessing(text):
    letters=re.sub("[^a-zA-Z]"," ",text)
    words=letters.lower().split()
    stops=set(stopwords.words("english"))
    meaningful_words=[w for w in words if not w in stops]
    return(" ".join(meaningful_words))
    
    
def create_cloud(text_to_draw):
    wordcloud = WordCloud().generate(text_to_draw)
    # Display the generated image:
    # the matplotlib way:
    import matplotlib.pyplot as plt
    plt.imshow(wordcloud)
    plt.axis("off")
    
    # take relative word frequencies into account, lower max_font_size
    wordcloud = WordCloud(max_font_size=40 ,relative_scaling=.5).generate(text_to_draw)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
risk_clean=[]
for row in range(0,len(risk_title)):
    risk_clean.append(preprocessing(risk_title[row]))
risk_clean=[x.encode('UTF8') for x in risk_clean]

risk_SN=[str(d[0]) for d in risk]
risk=zip(risk_SN,risk_clean)

risk_clean_str=" ".join(risk_clean)
create_cloud(risk_clean_str)

risk=pd.DataFrame(risk) 
risk_labelled=[]
for row in risk.iterrows():
    index, data = row
    risk_labelled.append(data.tolist())

string=["died","Died","Dies","Killed","killed","crushed","dead","Dead","Crushed","electrocuted","fatally","dies","Electric Shock"]    
#for i in str:   
match=[s for s in risk_labelled if any(xs in s[1] for xs in string)]

output = pd.DataFrame(match)
output.to_csv( "match.csv", index=False, quoting=3 )

for item in string:
   for d in risk_labelled:
    if item in d[1]:
        risk_labelled.append("death")
    else:
        risk_labelled.append("injury")
output = pd.DataFrame(match,columns=["SN","Title"])
output.to_csv( "pred.csv", index=False, quoting=3 )
    
   
    

train=pd.read_excel("osha-labelled.xlsx")
col_list=['Abstract','Occupation']
train=train[col_list]

labelled=[]
for row in train.iterrows():
    index,data=row
    labelled.append(preprocessing(data[0]))
    
train_label=train["Occupation"]
train_label=train_label.tolist()
train_labelled=zip(labelled,train_label)
     
    
vectorizer=TfidfVectorizer(min_df=1)
abs_train=vectorizer.fit_transform(labelled)



testdata=pd.read_excel("osha-labelled-test.xlsx")
testdata=testdata[testdata.Abstract!="InspectionOpen DateSICEstablishment Name"]

testdata = testdata.drop(testdata[(testdata.SN==202658258)|(testdata.SN==14240477)|(testdata.SN==200536050)|(testdata.SN==14372981)].index)

col_list=['Abstract']
test=testdata[col_list]

test_abs=[]
for row in test.iterrows():
    index,data=row
    test_abs.append(preprocessing(data[0]))
    
    
test_features=vectorizer.transform(test_abs)

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
dt = train_dtc(abs_train,train_label)
predDT = dt.predict(test_features)

output = pd.DataFrame( data={"id":testdata["SN"], "Occupation":predDT} )
output.to_csv( "pred.csv", index=False, quoting=3 )

# Print the classification rate


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
    
sv = train_svm(abs_train, train_label)
predSVM = sv.predict(test_features)

# Print the classification rate
print(sv.score(X_test, y_test)) 

output = pd.DataFrame( data={"id":testdata["SN"], "Occupation":predDT} )
output.to_csv( "pred.csv", index=False, quoting=3 )

 
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %% import dataset - UCI Heart Disease 
data = pd.read_csv('heart.csv')

# dataset informations
data.info()

# null-check
data.isna().sum()

# print first 5 rows
data.head(5)

# %%
X = data.drop(['target'], axis=1)
y = data['target']

# %% get categorical/continous variables
cat_val = []
cont_vals = []

for column in X.columns:
    if len(data[column].unique()) <= 10:
        cat_val.append(column)
    else:
        cont_vals.append(column)
X = pd.get_dummies(X, columns=cat_val)

X.head(5)

# %% feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
data[cont_vals] = sc.fit_transform(data[cont_vals])
scores = []
# %% split data into train/test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

# %% support vector classifier
from sklearn.svm import SVC
svc_classifier = SVC(kernel = 'rbf')
svc_classifier.fit(X_train, y_train)

y_pred = svc_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_svc = confusion_matrix(y_test, y_pred)

score_svc = round(svc_classifier.score(X_test, y_test)*100,2)

scores.append(score_svc)

# %% naive bayes classifier
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

y_pred = naive_bayes.predict(X_test)

cm_nb = confusion_matrix(y_test, y_pred)

score_nb = round(naive_bayes.score(X_test, y_test)*100,2)

scores.append(score_nb)

# %% random forest classifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(y_test, y_pred)

score_rf = round(naive_bayes.score(X_test, y_test)*100,2)

scores.append(score_rf)
# %%

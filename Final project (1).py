#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
#from sklearn.metrics import det_curve
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# First I load in the data:

# In[3]:


nba_elon = pd.read_csv("nbaallelo.txt", sep=",", header=0)
nba_elon


# In[4]:


nba_elon.describe()


# In[5]:


plt.figure(figsize=(10,5))
mask = nba_elon["is_playoffs"] == 0
pts_mean = nba_elon[mask].groupby(nba_elon[mask]["year_id"])["pts"].transform('mean')
plt.scatter(nba_elon[mask]["year_id"],nba_elon[mask]["pts"],color = "darkblue")
plt.plot(nba_elon[mask]["year_id"],pts_mean,color = "lightblue")
plt.title("Mean points scored per game vs each year")
regular = nba_elon[mask].describe()


# In[6]:


plt.figure(figsize=(10,5))
mask = nba_elon["is_playoffs"] == 1
pts_mean = nba_elon[mask].groupby(nba_elon[mask]["year_id"])["pts"].transform('mean')
plt.scatter(nba_elon[mask]["year_id"],nba_elon[mask]["pts"],color = "darkred")
plt.plot(nba_elon[mask]["year_id"],pts_mean,color = "red")
plt.title("Mean points scored per game vs each year")
playoffs = nba_elon[mask].describe()


# In[ ]:





# In[7]:


pd.set_option("display.max_rows", None, "display.max_columns", None)
print("\nRegular Season")
display(regular)
print("\nPlayoffs")
display(playoffs)
print("\nPlayoff - Regular Season")
display(playoffs - regular)


# In[8]:


regular = nba_elon[nba_elon["is_playoffs"] == 0]
regular = regular[["year_id","seasongame","opp_elo_i","elo_i","game_location","forecast","game_result"]]

regular = regular[regular["game_location"]!='N']

regular.game_result[regular.game_result == 'W'] = 1
regular.game_result[regular.game_result == 'L'] = 0

regular.game_location[regular.game_location == 'H'] = 1
regular.game_location[regular.game_location == 'A'] = 0

regular.dropna(axis = 0)
regular.head()


# In[9]:


print("There are", len(regular["year_id"]),"different samples")
X = regular.drop(columns = ["game_result"])
y = regular["game_result"]

display(X.head())
display(y.head())
# X = np.array(X)
# y = np.array(y)
y = y.astype('int')

print(X.shape,y.shape)


# In[25]:



roc_curves = []
log_losses = []
scores = []


# In[27]:



names = ["Nearest Neighbors"]
print(names)
#"Linear SVM", "RBF SVM"]
# , "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]

classifiers = [KNeighborsClassifier()]
#,SVC(kernel="linear", C=0.025),SVC(gamma=2, C=1)]
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=6),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]

parameters = {"n_neighbors":[5,10,50,100,1000,10000]}

# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)

datasets = [(X, y)]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=42)
    
    xx_min, xx_max = X_test[:, 3].min(), X_test[:, 3].max()
    yy_min, yy_max = X_test[:, 5].min(), X[:, 5].max()
    xx, yy = np.meshgrid(np.linspace(xx_min, xx_max, num = 100,endpoint=False),
                         np.linspace(yy_min, yy_max, num = 100,endpoint=False))
    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets),len(classifiers)+1,i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[0:10000, 3], X_train[0:10000, 5], c=y_train[0:10000], cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[0:10000, 3], X_test[0:10000, 5], c=y_test[0:10000], cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets),len(classifiers)+1,i)
        clf = GridSearchCV(clf, parameters)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
        y_pred = clf.predict(X_test)
        print(clf.cv_results_)
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(X_test)
            
        else:
            Z = clf.predict_proba(X_test)
            log_losses.append(log_loss(y_test,Z[:,1]))
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, Z[:,1], pos_label=1)
        roc_curves.append((fpr,tpr))
        
        # Put the result into a color plot
        Z = Z[0:10000][:,1]
        print(Z.shape)
        print(xx.shape)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[0:100:, 3], X_train[0:100:, 5], c=y_train[0:100], cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[0:100:, 3], X_test[0:100:, 5], c=y_test[0:100], cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()
figure.savefig("kNN", dpi=300)
plt.close(figure)


# In[ ]:


names = ["Linear SVM"]
print(names)
#, "RBF SVM"]
# , "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]

classifiers = [SVC()]
#,, C=0.025),SVC(gamma=2, C=1)]
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=6),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]

parameters = {"kernel":('linear', 'poly'),"C":[1,2]}

# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)

datasets = [(X, y)]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=42)
    
    xx_min, xx_max = X_test[:, 3].min(), X_test[:, 3].max()
    yy_min, yy_max = X_test[:, 5].min(), X[:, 5].max()
    xx, yy = np.meshgrid(np.linspace(xx_min, xx_max, num = 100,endpoint=False),
                         np.linspace(yy_min, yy_max, num = 100,endpoint=False))
    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets),len(classifiers)+1,i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[0:10000, 3], X_train[0:10000, 5], c=y_train[0:10000], cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[0:10000, 3], X_test[0:10000, 5], c=y_test[0:10000], cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets),len(classifiers)+1,i)
        clf = GridSearchCV(clf, parameters)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
        y_pred = clf.predict(X_test)
        print(clf.cv_results_)
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(X_test)
            
        else:
            Z = clf.predict_proba(X_test)
            log_losses.append(log_loss(y_test,Z[:,1]))
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, Z, pos_label=1)
        roc_curves.append((fpr,tpr))
        
        # Put the result into a color plot
        Z = Z[0:10000][:,1]
        print(Z.shape)
        print(xx.shape)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[0:100:, 3], X_train[0:100:, 5], c=y_train[0:100], cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[0:100:, 3], X_test[0:100:, 5], c=y_test[0:100], cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()
figure.savefig("SVC", dpi=300)
plt.close(figure)


# In[ ]:


roc_curves = np.array(roc_curves)
log_losses = np.array(log_losses)
scores = np.array(scores)
np.save("roc_curves",roc_curves)
np.save("log_losses",log_losses)
np.save("scores",scores)


# In[ ]:


names = ["Decision Tree"]
#, "RBF SVM"]
# , "Gaussian Process",
#          , "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]

classifiers = [DecisionTreeClassifier()]
#RandomForestClassifier(max_depth=5, n_estimators=10, max_features=6),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]

parameters = {"max_depth":[1,5,10]}

# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)

datasets = [(X, y)]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=42)
    
    xx_min, xx_max = X_test[:, 3].min(), X_test[:, 3].max()
    yy_min, yy_max = X_test[:, 5].min(), X[:, 5].max()
    xx, yy = np.meshgrid(np.linspace(xx_min, xx_max, num = 100,endpoint=False),
                         np.linspace(yy_min, yy_max, num = 100,endpoint=False))
    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets),len(classifiers)+1,i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[0:10000, 3], X_train[0:10000, 5], c=y_train[0:10000], cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[0:10000, 3], X_test[0:10000, 5], c=y_test[0:10000], cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets),len(classifiers)+1,i)
        clf = GridSearchCV(clf, parameters)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
        y_pred = clf.predict(X_test)
        print(clf.cv_results_)
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(X_test)
            
        else:
            Z = clf.predict_proba(X_test)
            log_losses.append(log_loss(y_test,Z[:,1]))
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, Z[:,1], pos_label=1)
        roc_curves.append((fpr,tpr))
        
        # Put the result into a color plot
        Z = Z[0:10000][:,1]
        print(Z.shape)
        print(xx.shape)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[0:100:, 3], X_train[0:100:, 5], c=y_train[0:100], cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[0:100:, 3], X_test[0:100:, 5], c=y_test[0:100], cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()


# In[ ]:


names = ["Random Forest"]
#, "RBF SVM"]
# , "Gaussian Process",
#          , , "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]

classifiers = [RandomForestClassifier()]
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]

parameters = {"n_estimators":[100,500,1000]}:

# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)

datasets = [(X, y)]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=42)
    
    xx_min, xx_max = X_test[:, 3].min(), X_test[:, 3].max()
    yy_min, yy_max = X_test[:, 5].min(), X[:, 5].max()
    xx, yy = np.meshgrid(np.linspace(xx_min, xx_max, num = 100,endpoint=False),
                         np.linspace(yy_min, yy_max, num = 100,endpoint=False))
    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets),len(classifiers)+1,i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[0:10000, 3], X_train[0:10000, 5], c=y_train[0:10000], cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[0:10000, 3], X_test[0:10000, 5], c=y_test[0:10000], cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets),len(classifiers)+1,i)
        clf = GridSearchCV(clf, parameters)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
        y_pred = clf.predict(X_test)
        print(clf.cv_results_)
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(X_test)
            
        else:
            Z = clf.predict_proba(X_test)
            log_losses.append(log_loss(y_test,Z[:,1]))
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, Z[:,1], pos_label=1)
        roc_curves.append((fpr,tpr))
        
        # Put the result into a color plot
        Z = Z[0:10000][:,1]
        print(Z.shape)
        print(xx.shape)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[0:100:, 3], X_train[0:100:, 5], c=y_train[0:100], cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[0:100:, 3], X_test[0:100:, 5], c=y_test[0:100], cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()


# In[ ]:


names =["QDA"]
#, "RBF SVM"]
# , "Gaussian Process",
#          , , "Neural Net", "AdaBoost",
#          "Naive Bayes", ]

classifiers = [QuadraticDiscriminantAnalysis()]
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     ]

parameters = {"reg_param":[0]}

# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)

datasets = [(X, y)]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=42)
    
    xx_min, xx_max = X_test[:, 3].min(), X_test[:, 3].max()
    yy_min, yy_max = X_test[:, 5].min(), X[:, 5].max()
    xx, yy = np.meshgrid(np.linspace(xx_min, xx_max, num = 100,endpoint=False),
                         np.linspace(yy_min, yy_max, num = 100,endpoint=False))
    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets),len(classifiers)+1,i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[0:10000, 3], X_train[0:10000, 5], c=y_train[0:10000], cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[0:10000, 3], X_test[0:10000, 5], c=y_test[0:10000], cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets),len(classifiers)+1,i)
        clf = GridSearchCV(clf, parameters)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
        y_pred = clf.predict(X_test)
        print(clf.cv_results_)
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(X_test)
            
        else:
            Z = clf.predict_proba(X_test)
            log_losses.append(log_loss(y_test,Z[:,1]))
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, Z[:,1], pos_label=1)
        roc_curves.append((fpr,tpr))
        
        # Put the result into a color plot
        Z = Z[0:10000][:,1]
        print(Z.shape)
        print(xx.shape)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[0:100:, 3], X_train[0:100:, 5], c=y_train[0:100], cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[0:100:, 3], X_test[0:100:, 5], c=y_test[0:100], cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()


# In[ ]:


names = ["Logistic Regression"]
#, "RBF SVM"]
# , "Gaussian Process",
#          , , "Neural Net", "AdaBoost",
#          "Naive Bayes", ]

classifiers = [LogisticRegression()]
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     ]

parameters = {"penalty":("l1", "l2", "elasticnet", "none")}

# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
# X += 2 * rng.uniform(size=X.shape)

datasets = [(X, y)]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=42)
    
    xx_min, xx_max = X_test[:, 3].min(), X_test[:, 3].max()
    yy_min, yy_max = X_test[:, 5].min(), X[:, 5].max()
    xx, yy = np.meshgrid(np.linspace(xx_min, xx_max, num = 100,endpoint=False),
                         np.linspace(yy_min, yy_max, num = 100,endpoint=False))
    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets),len(classifiers)+1,i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[0:10000, 3], X_train[0:10000, 5], c=y_train[0:10000], cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[0:10000, 3], X_test[0:10000, 5], c=y_test[0:10000], cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets),len(classifiers)+1,i)
        clf = GridSearchCV(clf, parameters)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
        y_pred = clf.predict(X_test)
        print(clf.cv_results_)
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(X_test)
            
        else:
            Z = clf.predict_proba(X_test)
            log_losses.append(log_loss(y_test,Z[:,1]))
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test, Z[:,1], pos_label=1)
        roc_curves.append((fpr,tpr))
        
        # Put the result into a color plot
        Z = Z[0:10000][:,1]
        print(Z.shape)
        print(xx.shape)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[0:100:, 3], X_train[0:100:, 5], c=y_train[0:100], cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[0:100:, 3], X_test[0:100:, 5], c=y_test[0:100], cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()


# In[ ]:





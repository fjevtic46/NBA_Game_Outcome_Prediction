#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

get_ipython().run_line_magic('matplotlib', 'inline')


# First I load in the data:

# In[2]:


nba_elon = pd.read_csv("nbaallelo.txt", sep=",", header=0)
nba_elon


# In[3]:


nba_elon.describe()


# In[4]:


plt.figure(figsize=(10,5))
mask = nba_elon["is_playoffs"] == 0
pts_mean = nba_elon[mask].groupby(nba_elon[mask]["year_id"])["pts"].transform('mean')
plt.scatter(nba_elon[mask]["year_id"],nba_elon[mask]["pts"],color = "darkblue")
plt.plot(nba_elon[mask]["year_id"],pts_mean,color = "lightblue")
plt.title("Mean points scored per game vs each year")
regular = nba_elon[mask].describe()


# In[5]:


plt.figure(figsize=(10,5))
mask = nba_elon["is_playoffs"] == 1
pts_mean = nba_elon[mask].groupby(nba_elon[mask]["year_id"])["pts"].transform('mean')
plt.scatter(nba_elon[mask]["year_id"],nba_elon[mask]["pts"],color = "darkred")
plt.plot(nba_elon[mask]["year_id"],pts_mean,color = "red")
plt.title("Mean points scored per game vs each year")
playoffs = nba_elon[mask].describe()


# In[ ]:





# In[6]:


pd.set_option("display.max_rows", None, "display.max_columns", None)
print("\nRegular Season")
display(regular)
print("\nPlayoffs")
display(playoffs)
print("\nPlayoff - Regular Season")
display(playoffs - regular)


# In[7]:


regular = nba_elon[nba_elon["is_playoffs"] == 0]
regular = regular[["year_id","seasongame","opp_elo_i","elo_i","game_location","forecast","game_result"]]

regular = regular[regular["game_location"]!='N']

regular.game_result[regular.game_result == 'W'] = 1
regular.game_result[regular.game_result == 'L'] = 0

regular.game_location[regular.game_location == 'H'] = 1
regular.game_location[regular.game_location == 'A'] = 0

regular.dropna(axis = 0)
regular.head()


# In[8]:


print("There are", len(regular["year_id"]),"different samples")
X = regular.drop(columns = ["game_result"])
y = regular["game_result"]

display(X.head())
display(y.head())
X = np.array(X)
y = np.array(y)
y = y.astype('int')

print(X.shape,y.shape)


# In[9]:


h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,random_state=1, n_clusters_per_class=1)
# rng = np.random.RandomState(2)
#X += 2 * rng.uniform(size=X.shape)
print(X.shape)
print(y.shape)



datasets = [(X, y)]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=42)
    
    grid = []
    for i in range(X.shape[1]):
        i_min, i_max = X[:, i].min() - .5, X[:, i].max() + .5
        grid.append(np.arange(i_min, i_max, h))
    x1,x2,x3,x4,x5,x6 = grid[0],grid[1],grid[2],grid[3],grid[4],grid[5] 
    x1,x2,x3,x4,x5,x6 = np.meshgrid(x1,x2,x3,x4,x5,x6)

    #print(xx[0],xx.shape,yy[0],yy.shape)

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 3], X_train[:, 4], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 3], X_test[:, 4], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(x4.min(), x4.max())
    ax.set_ylim(x5.min(), x5.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        print(X_train.shape)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[x1.ravel(),x2.ravel(),x3.ravel(),x5.ravel(), x5.ravel(),x6.ravel()])
        else:
            Z = clf.predict_proba(np.c_[x1.ravel(),x2.ravel(),x3.ravel(),x5.ravel(), x5.ravel(),x6.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(x1.shape)
        ax.contourf(x4, x5, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 3], X_train[:, 4], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 3], X_test[:, 4], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(x4.min(), x4.max())
        ax.set_ylim(x5.min(), x5.max())
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





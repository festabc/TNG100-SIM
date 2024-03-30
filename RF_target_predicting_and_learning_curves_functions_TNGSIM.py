#!/usr/bin/env python
# coding: utf-8

# ## Target Predicting Function: Random Forest Regressor (RF), and its learning curves 

# In[1]:


import time
import numpy as np
import pandas as pd

import galsim #install with conda install -c conda_forge galsim

import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.cm as cm
import matplotlib.colors as norm
from matplotlib.gridspec import SubplotSpec
import seaborn as sns

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline #This allows one to build different steps together
from sklearn.preprocessing import StandardScaler, RobustScaler

from tqdm import tqdm 


# In[2]:


from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring = 'r2', scale = False):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    
    fig, ax = plt.subplots(figsize=(7, 5))
#     plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("# of training examples",fontsize = 14)
 
    ax.set_ylabel(r"$R^{2}$ score",fontsize = 14)
    
    if (scale == True):
        scaler = sklearn.preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring = scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
#    plt.grid()

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="b")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="b",
             label="Training score from CV")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test score from CV")

    ax.legend(loc="best",fontsize = 12)
    return fig


# In[3]:


def run_random_search_pipeline(X, y):
    
    # Best params, best score: 0.8081 {'bootstrap': True, 'max_depth': None, 'max_features': 'auto', 'min_samples_leaf': 2, # 'min_samples_split': 2, 'n_estimators': 500}
    n_estimators = [500]
    grid = {'bootstrap': [True],
                   'max_depth': [None],
                   'max_features': ['auto'],
                   'min_samples_leaf': [2],
                   'min_samples_split': [2],
                   'n_estimators': n_estimators}

#     n_estimators = [20,50,100]
#     grid = {'bootstrap': [True],
#                    'max_depth': [5, 10, None],
#     #                'max_features': ['auto', 'sqrt'],
#                    'min_samples_leaf': [2],
#     #                'min_samples_split': [2, 5, 10],
#                    'n_estimators': n_estimators}
    
#     pipeline = make_pipeline(RandomForestRegressor())
    model=RandomForestRegressor(random_state=0)

    reg_RF = GridSearchCV(model, grid, cv = KFold(n_splits=3, shuffle=True),                          verbose = 3, n_jobs = -1, return_train_score=True)
    reg_RF.fit(X, y)
#     estimator.get_params().keys()
    scores_lim = pd.DataFrame(reg_RF.cv_results_)

    scores_lim = scores_lim[['params','mean_test_score','std_test_score','mean_train_score', 'mean_fit_time']].sort_values(by = 'mean_test_score', ascending = False)
    
    return scores_lim['mean_test_score']


# In[7]:


def target_predicting_RF_function(X, y, group_and_title):
    '''This function takes in a dataframe of X(features) and y (a target) and outputs :
    
    1) Predictions of the target through the Random Forest Regressor ML model. 
    
    2) Feature Ranking using RF built-in ranking functions MDI and Feature Permutation
    
    3) The train and test Learning Curve for the best RF function.
    
    group_and_title should be a string that defines the galaxy morphplogy and the title of the plot'''
    
        
    regr_RF = RandomForestRegressor(random_state=0)

#     Best params, best score: 0.7762 {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 2, 'n_estimators': 100}
# RF Accuracy: 0.78 (+/- 0.01)

#     # number of trees in random forest
#     n_estimators = [500]
#     # create random grid
#     grid = {'bootstrap': [True],
#                    'max_depth': [None],
#     #                'max_features': ['auto', 'sqrt'],
#                    'min_samples_leaf': [2],
#     #                'min_samples_split': [2, 5, 10],
#                    'n_estimators': n_estimators}
    
    # number of trees in random forest
    n_estimators = [50,100, 500]
    # create random grid
    grid = {'bootstrap': [True],
                   'max_depth': [5, 10, None],
    #                'max_features': ['auto', 'sqrt'],
                   'min_samples_leaf': [1, 2, 4],
    #                'min_samples_split': [2, 5, 10],
                   'n_estimators': n_estimators}
    
    # Grid search of parameters
    rfr_grid = GridSearchCV(estimator = regr_RF, param_grid = grid, cv = KFold(n_splits=5, shuffle=True),                          verbose = 1, n_jobs = -1, return_train_score=True)

    rfr_grid.fit(X, y)

    print('Best params, best score:', "{:.4f}".format(rfr_grid.best_score_),         rfr_grid.best_params_)
    # define trainscore according to best model
    best_RF=rfr_grid.best_estimator_ # The index (of the cv_results_ arrays) which corresponds to the best candidate parameter setting.

    best_RF_trainscore=rfr_grid.cv_results_['mean_train_score'][rfr_grid.best_index_]

    # predict y (galaxy sizes) by using 5-fold cross-validation
    y_pred_RF = cross_val_predict(best_RF, X, y, cv = KFold(n_splits=5, shuffle=True, random_state=10))

    # find prediction scores of each of the cross validation fold
    scores_RF = cross_val_score(best_RF, X, y, cv = KFold(n_splits=5, shuffle=True, random_state=10))
    print("RF Accuracy: %0.2f (+/- %0.2f)" % (scores_RF.mean(), scores_RF.std() * 2))
    
    # Plot the figure
    fig_prediction, ax = plt.subplots(figsize=(5, 5))

    label = 'Train score={} \n Test score={}'.format(round(best_RF_trainscore, 2),round(scores_RF.mean(), 2) )
    ax = plt.subplot()
    ax.scatter(y, y_pred_RF, s=3, marker='.', alpha=0.7, label=label)
#     ax.set_xlim([0.0,100])
#     ax.set_ylim([0.0,100])
#     ax.plot([0.0, 170], [0.0, 170], color = 'black', linewidth = 2)
    ax.axis([-0.25,1.5, -0.25,1.5])
    ax.plot([-1.0, 2.0], [-1.0, 2.0], color = 'black', linewidth = 2)
    ax.set_title('TNG-SIM Raw Dataset \n' + r'RF w $log_{10}$(all features)', fontsize=14)
    ax.set_xlabel(r'True $log_{10}Rstar$', fontsize=12)
    ax.set_ylabel(r'Predicted $log_{10}Rstar$', fontsize=12)
    ax.legend(loc='lower right')

    fig_prediction.tight_layout()
    # plt.savefig('Prediction_vs_True.jpeg', dpi=500)

    plt.show()
        
    # Extract important features using built-in functions
    start_time = time.time()
    importances = rfr_grid.best_estimator_.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfr_grid.best_estimator_.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


    feature_names = X.columns

    forest_importances = pd.Series(importances, index=feature_names)

    fig_perm_import, [ax1, ax2] = plt.subplots(1,2, figsize=(15,5))
    fig_perm_import.suptitle('TNG-SIM Feature Importances  ', fontsize=16, fontweight='bold')

    forest_importances.plot.bar(yerr=std, ax=ax1)
    ax1.set_title("Feature importances using Mean Decrease in Impurity \n " + r'Predict $log_{10}$Rstar using $log_{10}$(all features)')
    ax1.set_ylabel("Mean decrease in impurity")

    ### Feature Importances Permutation

    start_time = time.time()
    result = permutation_importance(
        rfr_grid.best_estimator_, X, y, n_repeats=10, random_state=0, n_jobs=-1
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    forest_importances.plot.bar(yerr=result.importances_std, ax=ax2)
    ax2.set_title("Feature importances using Permutation Importances ")
    ax2.set_ylabel("Mean accuracy decrease")
    fig_perm_import.tight_layout()
    # plt.savefig("Feature_Importances_v2.jpeg", dpi=500)

    plt.show()

    df_perm_import=pd.DataFrame(forest_importances).sort_values(0, ascending=False)
#     df_perm_import.to_csv('RF built in Feature Importances Permutation.csv', index=False)

    lc_fig = plot_learning_curve(best_RF, group_and_title, X, y, ylim=(0.0, 1.0), train_sizes = np.array([0.05,0.1,0.2,0.5,1.0]), scoring='r2', cv = KFold(n_splits=5, shuffle=True))
    plt.show()
    # plt.savefig('LC_SVR_Group1.jpeg', dpi=500)

   
    return y_pred_RF, fig_prediction, fig_perm_import, df_perm_import, lc_fig, round(best_RF_trainscore, 2),round(scores_RF.mean(), 2)


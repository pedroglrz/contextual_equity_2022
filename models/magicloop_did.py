
# Import Statements
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
import random
import time
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.model_selection import ParameterGrid
import warnings
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
import argparse

def topk_predict(y_proba, k):
    #predict based on topk threshold
    y_topk_preds = y_proba.copy()
    top_proba = list(y_proba.argsort()[-k:][::-1])
    bot_proba = list(y_proba.argsort()[:-k])
    y_topk_preds[top_proba] =1.
    y_topk_preds[bot_proba] =0.
    return y_topk_preds

def get_subsets(l):
    subsets = []
    for i in range(1, len(l) + 1):
        for combo in itertools.combinations(l, i):
            subsets.append(list(combo))
    return subsets

def bool_intersetion_dict(subset,super_set):
    intersection_dict = {}
    for item in super_set:
        if item in subset:
            intersection_dict[item] = 1
        else:
            intersection_dict[item] = 0
    return intersection_dict

def join_subset_features(subset,feature_set_dict):
    subset_features = []    
    for feature_set in subset:
#         print(feature_set)
        subset_features.append(feature_set_dict[feature_set])
    subset_features = [item for sublist in subset_features for item in sublist]
    return(subset_features)


#define grids and models
clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
    'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
    'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
    'LR': LogisticRegression(penalty='l1', C=1e5),
    'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
    'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
    'NB': GaussianNB(),
    'DT': DecisionTreeClassifier(),
    'SGD': SGDClassifier(loss="hinge", penalty="l2"),
    'KNN': KNeighborsClassifier(n_neighbors=3) 
        }

def select_grid(grid_name):
    large_grid = { 
                    'RF':{'n_estimators': [1,10,100], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
                    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
                    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
                    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
                    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
                    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
                    'NB' : {},
                    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': [None],'min_samples_split': [2,5,10]},
                    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
                    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
                           }
    small_grid = { 
        'RF':{'n_estimators': [10, 100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs':[-1]},
        'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10],'solver':['liblinear']},
        'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
        'ET': { 'n_estimators': [100, 10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs':[-1]},
        'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
        'GB': {'n_estimators': [100, 10000], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
        'NB' : {},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': [None],'min_samples_split': [2,5,10]},
        'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
        'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
               }

    medium_grid = { 
        'RF':{'n_estimators': [10,100,200], 'max_depth': [5,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs':[-1]},
        'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10,100,1000],'solver':['liblinear']},
        'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
        'ET': { 'n_estimators': [100, 10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs':[-1]},
        'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
        'GB': {'n_estimators': [100, 10000], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
        'NB' : {},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': [None],'min_samples_split': [2,5,10]},
        'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
        'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
               }       

    medium_grid2 = { 
        'RF':{'n_estimators': [10,100,200,1000], 'max_depth': [5,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs':[-1]},
        'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10,100,1000],'solver':['liblinear']},
        'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
        'ET': { 'n_estimators': [100, 10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs':[-1]},
        'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
        'GB': {'n_estimators': [100, 10000], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
        'NB' : {},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': [None],'min_samples_split': [2,5,10]},
        'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
        'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
               }                              

    test_grid = { 
        'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10], 'n_jobs': [-1]},
        'LR': { 'penalty': ['l1'], 'C': [0.01],'solver':['liblinear']},
        'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
        'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10], 'n_jobs': [-1]},
        'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
        'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
        'NB' : {},
        'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': [None],'min_samples_split': [10]},
        'SVM' :{'C' :[0.01],'kernel':['linear']},
        'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
               }
    
    if grid_name == "large_grid":
        return large_grid
    elif grid_name == "small_grid":
        return small_grid
    elif grid_name == "medium_grid":
        return medium_grid  
    elif grid_name == "medium_grid2":
        return medium_grid2                      
    else:
        return test_grid   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required = True,
                        help = "twitter user file in data directory")
    parser.add_argument("--data", type=str, required = True,
                        help = "twitter user file in data directory")                    
    parser.add_argument("--grid_size", type=str, required = True,
                        help = "where to write user's errors when collecting tweets")

    args = parser.parse_args()

    data_df = pd.read_csv(os.path.join(args.data_dir,args.data))

    #filter to complete applications
    data_complete_df = data_df.copy()
    data_complete_df = data_complete_df[data_complete_df.std_decision.isin(['matriculated','deny'])]
    
    #define feature_set
    demo_features = [] #Race, cultureal_id, sex, hispanic, #income, parent income
    academic_features = [] #gpas, non-nyu-stem,
    application_features = [] # stubmitted dates, started dates, missing itmes
    school_features = [] #schools


    #define feature_set
    for col in data_complete_df.columns:
        #demo features
        if col.startswith('oh_race'):
            demo_features.append(col)     
        elif col.startswith('oh_cultural_id'):
            demo_features.append(col)
        elif col.startswith('oh_sex'):
            demo_features.append(col)
        elif col.startswith('oh_hispanic'):
            demo_features.append(col)
        elif col.startswith('oh_income'):
            demo_features.append(col)
        elif col.startswith('oh_parent'):
        #academic feature
            demo_features.append(col)
        elif col.startswith('oh_gpa'):
            academic_features.append(col)
        elif col.startswith('std_imputed_gpa'):
            academic_features.append(col)        
        elif col.startswith('oh_non'):
            academic_features.append(col)
        #application features        
        elif col.startswith('oh_missing'):
            application_features.append(col)  
        elif col.startswith('std_imputed_app'):
            application_features.append(col)        
    #     school features
        elif col.startswith('oh_school_'):
            school_features.append(col)    

    feature_set_dict = {
                        'demo_features': demo_features,
                        'academic_features': academic_features,
                        'application_features': application_features,
                        'school_features': school_features,
                        }

    models_to_run = ['RF','LR','DT']
    grid = select_grid(args.grid_size)



    feature_set_list = list(feature_set_dict.keys())
    feature_subsets = get_subsets(feature_set_list)
    data_complete_df['oh_decision'] = (data_complete_df.std_decision == 'matriculated')*1

    train, test = train_test_split(data_complete_df, test_size=0.2, random_state=1, stratify = data_complete_df.oh_decision)

    metrics_dict_list = []
    target_df = train.oh_decision
    y = np.array(target_df)


    for model_i,clf in enumerate([clfs[x] for x in models_to_run]):
        print("---Running {} Loop---".format(models_to_run[model_i]))
        parameter_values = grid[models_to_run[model_i]]  #change grid here
        for p in ParameterGrid(parameter_values):
            print("Running Parameters {}".format(p))
            clf.set_params(**p)
            
            #loop through feature set combinations
            for subset in feature_subsets:
                print("Running Subset {}".format(subset))
                #create feature list from subsets
                indicator_dict = bool_intersetion_dict(subset, feature_set_list)
                subset_features = join_subset_features(subset, feature_set_dict)
                feature_df = train[subset_features]
                
                #define feature matrix
                X = np.array(feature_df)
                
                #define k-fold
                rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,random_state=36851234)
                rskf.get_n_splits(X, y)
                print("Starting Repeated Cross Validation (5x5)")
                for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
                    metrics_dict = {}

                    #train test split for xval
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    #fit model, make predictions
                    clf.fit(X_train,y_train)
                    y_proba = clf.predict_proba(X_test)[:,1]
                    y_preds = clf.predict(X_test)

                    #predict based on topk threshold
                    y_topk_preds = topk_predict(y_proba, k = y_test.sum())

                    #topk metrics
                    metrics_dict['clf'] = models_to_run[model_i]
                    metrics_dict['param'] = str(p)
                    
                    for feature_set in feature_set_list:
                        metrics_dict[feature_set] = indicator_dict[feature_set]

                    metrics_dict['xv_split'] = i
                    metrics_dict['topk_precision'] = precision_score(y_test,y_topk_preds)
                    metrics_dict['topk_recall'] = recall_score(y_test,y_topk_preds)            
                    metrics_dict['random_topk_precision'] = y_test.sum()/len(y_test)

                    #default thresholding metrics
        #             precision, recall, fscore, _ = precision_recall_fscore_support(y_test,y_preds,average='binary')
                    metrics_dict['precision'] = precision_score(y_test,y_preds,average='binary')
                    metrics_dict['recall'] = recall_score(y_test,y_preds,average='binary') 
        #             metrics_dict['fscore'] = fscore
                    metrics_dict['auc'] = roc_auc_score(y_test, y_proba)
                    metrics_dict['model'] = clf
                    metrics_dict_list.append(metrics_dict)
                    
                print("Completing Repeated Cross Validation for Loop")

    print("WRITING OUTPUTS")
    metrics_df = pd.DataFrame(metrics_dict_list)  
    metrics_df.to_csv(os.path.join(args.data_dir,"{0}-100_{1}_{2}".format(args.grid_size,'result-metrics',args.data)), index=False)
    test.to_csv(os.path.join(args.data_dir,"{0}-100_{1}_{2}".format(args.grid_size,'test-set',args.data)), index=False)
    train.to_csv(os.path.join(args.data_dir,"{0}-100_{1}_{2}".format(args.grid_size,'train-set',args.data)), index=False)
    data_complete_df.to_csv(os.path.join(args.data_dir,"{0}-100_{1}_{2}".format(args.grid_size,'full-set',args.data)), index=False)
    print("FINISHED WRITING OUTPUTS")


if __name__ == '__main__':
    main()


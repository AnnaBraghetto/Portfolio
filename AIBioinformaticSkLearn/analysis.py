import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier


class Analysis():

    def __init__(self,X_train,y_train,X_test,y_test):
        '''Initialization of class Analysis
        Input:
        X_train = np.array of features for training set
        y_train = np array of labels for training set
        X_test = np array of features for test set
        y_test = np array of labels for test set
        '''

        #save input features
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        #define classifiers
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(probability=True),
            'Random Forest': RandomForestClassifier(),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(),
            'XGBoost':XGBClassifier(eval_metric='logloss', verbosity=0)
            }

    
    def cross_validation(self,n_splits=5):
        '''Method to perform cross validation
        Input:
        n_splits = int number of splits in cv
        Output:
        '''

        #define stratified k-fold cross-validation
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
        
        #store evaluation results
        results = []
        
        #evaluate each model with a pipeline
        for name, model in self.models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', model)
            ])
            
            scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=cv, scoring='accuracy')
            results.append({
                'Model': name,
                'Mean Accuracy': np.mean(scores),
                'Std Accuracy': np.std(scores)
            })
        
        #create results dataframe
        df_results = pd.DataFrame(results).sort_values(by='Mean Accuracy', ascending=False)
        print(df_results)
    
        #show best model
        self.best_model_name = df_results.iloc[0]['Model']
        print(f'Best Model: {self.best_model_name}')

    
    def best_model_analysis(self):
        '''Method to perform analysis with best model
        Input:
        Output:
        '''

       
        pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', self.models[self.best_model_name])
            ])

        #train the model
        pipeline.fit(self.X_train, self.y_train)
        
        #make predictions on train set
        y_train_pred = pipeline.predict(self.X_train)
        #make predictions on test set
        y_test_pred = pipeline.predict(self.X_test)

        print('Train accuracy =', accuracy_score(self.y_train, y_train_pred))
        print('Test accuracy =', accuracy_score(self.y_test, y_test_pred))

        #compute confusion matrix
        cm = confusion_matrix(self.y_test,y_test_pred)
        
        #plot confusion matrix
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
        plt.xlabel('predicted')
        plt.ylabel('actual')
        plt.show()
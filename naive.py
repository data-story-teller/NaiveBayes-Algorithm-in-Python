#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:39:27 2020

@author: mahade
"""

#import libraries
import random
import math
import pandas as pd
from collections import defaultdict
from data_label import data_labelling


"""
This is the class for data pre-processing. Making the 5 fold, 
Train-test Split, Data label encryption, Dropping useless columns are executed in this class
"""
class DataPreprocessing():
    #initializer for this class
    def __init__(self, data):
        self.data = data_labelling(data) #data is sent off to the 'data_labelling function'
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.length = None
        self.unique_class_values_prob = None
        self.unique_class_count = None
        self.train_data = None
        
        self.train_data_folds = None
        self.test_data_folds = None
        
        
    #The 5 folds are made here. Each of the train and test folds are saved into python dictionary
    def make_folds(self, k=5): 
        train_data_folds = {}
        test_data_folds = {}
        for i in range(0,k):
            dataCopy=self.data.copy()
            totalNoOfRows = len(self.data.index)
            numTrainRows = math.floor(0.80*totalNoOfRows)
            trainRows= self._Rand(totalNoOfRows,numTrainRows)
            dfTrain=pd.DataFrame(dataCopy.iloc[trainRows])
            train_data_folds['fold' + str(i)] = dfTrain    
            dfTest=dataCopy.drop(dfTrain.index)
            test_data_folds['fold' + str(i)] = dfTest
            
        self.train_data_folds = train_data_folds
        self.test_data_folds = test_data_folds
    
    """Internal function being used for..."""
    def _Rand(self,limit, num): 
        return (random.sample(range(limit-1),num )) 
    
    #Trin-Test splits
    def train_test_split(self, fold, pred_col):
        self.train_data = self.train_data_folds[fold]
        self.test_data = self.test_data_folds[fold]
        self.length = len(self.train_data[pred_col])
        self.unique_class_values_prob = dict(self.train_data[pred_col].value_counts()/self.length)
        self.unique_class_count = dict(self.train_data[pred_col].value_counts())
        self.X_train, self.y_train, self.X_test, self.y_test = self.train_data.drop([pred_col], axis=1).iloc[:, :], pd.DataFrame(self.train_data.iloc[:, self.train_data.columns.get_loc(pred_col)], columns = [pred_col]), self.test_data.drop([pred_col], axis=1).iloc[:, :], pd.DataFrame(self.test_data.iloc[:, self.test_data.columns.get_loc(pred_col)], columns = [pred_col])
        
        return self.X_train, self.X_test, self.y_train, self.y_test


"""
The NaiveBayes classifier algorithm. All the mathematical calculations are done within 
this block for example initial probabilities, predictions, accuracy measure etc.
"""
class NaiveBayes(DataPreprocessing):
    """
    Initializer for this class. It can be initialized with or without the 'fold' that we want to calculate the 
    classifier for. When an instance is created for this class the first time, it doesn't have a fold. However, internal functions are used to 
    iterate over different folds to calculate an avg score.
    """
    def __init__(self, dp, fold=None):
        if fold:
            self.fold = fold
            self.X_train, self.X_test, self.y_train, self.y_test = dp.train_test_split(self.fold, 'class') 
            self.length = dp.length
            self.unique_class_values_prob = dp.unique_class_values_prob
            self.unique_class_count = dp.unique_class_count
            self.train_data = dp.train_data
        else:
            pass
    
    """
    Count table from which probabilities will be calculated. This an internal function
    """
    def _make_count_table(self, X_train, X_test, y_train, y_test):
        dict1 = defaultdict(list)
        dict2 = defaultdict(list)
        table_dict = defaultdict(list)
        row_index = []
        
        for col in X_train.columns:
            for val in X_train[col].unique():
                dict1[col].append(val)
        
        for col in y_train.columns:
            for val in y_train[col].unique():
                dict2[col].append(val)
        
        
        for key1, val1 in dict2.items():
            for vals1 in val1:
                for key2, val2 in dict1.items():
                    for vals2 in val2:
                        table_dict[vals1].append(len(self.train_data[(self.train_data[key1] == vals1) & (self.train_data[key2] == vals2)][[key2]]))
                        row_index.append(vals2)
        
        prob_table = pd.DataFrame(table_dict, index = row_index[:int(len(row_index)/len(set(y_train['class'])))])
        prob_table['Total'] = prob_table.apply(lambda row: row.e + row.p, axis = 1)
        return prob_table
        
    
    """calculate class probabilities and feature probabilities"""
    def calc_initial_probabilities(self):
        prob_dictionary = {}
        conditional_probs = {}
        prob_table = self._make_count_table(self.X_train,self.X_test,self.y_train,self.y_test)
        
        for index, row in prob_table.iterrows():
            prob_dictionary[index] = row['Total']/self.length
            
        for val in self.unique_class_count:
            for index, row in prob_table.iterrows():
                conditional_probs[index + '|' + val] = row[val]/self.unique_class_count[val]
        
        return prob_dictionary, self.unique_class_values_prob, conditional_probs
        
    
    """Calculate the confusion matrix, accuracy and sensitivity for each fold"""
    def cross_validation(self):
        confusion_matrix = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        y_pred = self.predict()
        y_test = [list(value) for key,value in self.y_test.iteritems()]
        y_test = y_test[0]
        
        for ground_truth, predicted in zip(y_test, y_pred):
            if ground_truth == 'e' and predicted == 'e':
                confusion_matrix['tp'] = confusion_matrix['tp']+1
            if ground_truth == 'e' and predicted == 'p':
                confusion_matrix['fn'] = confusion_matrix['fn']+1
            if ground_truth == 'p' and predicted == 'e':
                confusion_matrix['fp'] = confusion_matrix['fp']+1
            if ground_truth == 'p' and predicted == 'p':
                confusion_matrix['tn'] = confusion_matrix['tn']+1
        
        
        sensitivity = confusion_matrix['tp']/(confusion_matrix['tp'] + confusion_matrix['fn'])
        specificity = confusion_matrix['tn']/(confusion_matrix['tn'] + confusion_matrix['fp'])
        accuracy = (confusion_matrix['tp'] + confusion_matrix['tn'])/(confusion_matrix['tp'] + confusion_matrix['tn'] + confusion_matrix['fp'] + confusion_matrix['fn'])
        
        return confusion_matrix, sensitivity,specificity, accuracy  
        
        
    
    """predict test data and returns predicted values in a list"""
    def predict(self):
        self.prob_dictionary, self.unique_class_values_prob, self.conditional_probs = self.calc_initial_probabilities()
        unique_class_vals = list(set(self.y_test['class']))
        equation_ratio = 1
        final_prob = {}
        y_pred = []
        
        for index, rows in self.X_test.iterrows():
            for vals in unique_class_vals:
                for feature in rows.values:
                    equation_ratio *= self.conditional_probs[feature + '|' + vals]/self.prob_dictionary[feature]
                    
                final_prob[vals] = equation_ratio*self.unique_class_values_prob[vals]
                equation_ratio = 1
            
            total = sum([val for key, val in final_prob.items()])
            normalized_probability = {k: v / total for k, v in final_prob.items()}
            y_pred.append(list(normalized_probability.keys())[list(normalized_probability.values()).index(max([val for key, val in normalized_probability.items()]))])
            final_prob = {}
        
        return y_pred
                
     
    """fit the model and calculate average score to validate the model""" 
    def fit_model(self, dp):
        folds = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']
        confusion_matrix = []
        sensitivity = []
        specificity = []
        accuracy = []
        for fold in folds:
            print('----', fold, '----')
            nb = NaiveBayes(dp, fold)
            cm, st,sp, acc = nb.cross_validation()
            confusion_matrix.append(cm)
            sensitivity.append(st)
            specificity.append(sp)
            accuracy.append(acc)
            
            print("Confusion Matrix: ", cm)
            print("Sensitivity: ", st)
            print("Specificity: ", sp)
            print("Accuracy: ", acc)
            
            print('\n')
        
        print("Final result:---------------------------------------------")
        print("Average Sensitivity: ", sum(sensitivity)/len(sensitivity))
        print("Average Specificity: ", sum(specificity)/len(specificity))
        print("Average Accuracy: ", sum(accuracy)/len(accuracy))
        
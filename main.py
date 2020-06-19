#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Import libraries"""
import pandas as pd
import naive

"""Load data into pandas dataframe"""
df = pd.read_csv("mushrooms.csv")

"""Create an object of the class for pre-processing"""
dp = naive.DataPreprocessing(df)

"""make folds and initialize all the attributes"""
dp.make_folds()

"""create an instance for the classifier"""
nb = naive.NaiveBayes(dp)

"""fit the model on the pre-processed data and measure model performance on with cross validation"""
nb.fit_model(dp)
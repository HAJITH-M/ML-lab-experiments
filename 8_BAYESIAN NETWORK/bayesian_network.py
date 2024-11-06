#pip install -r requirements.txt
import numpy as np
import csv
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

heartDisease = pd.read_csv('./heart.csv')
heartDisease = heartDisease.replace('?', np.nan)

print('Few examples from the dataset are given below')
print(heartDisease.head())

model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'),
                      ('sex', 'trestbps'), ('exang', 'trestbps'), ('trestbps', 'heartdisease'),
                      ('fbs', 'heartdisease'), ('heartdisease', 'restecg'),
                      ('heartdisease', 'thalach'), ('heartdisease', 'chol')])

print('\n Learning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

print('\n Inferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)

print('\n 1. Probability of HeartDisease given Age=30')
unique_ages = heartDisease['age'].unique()

selected_age = unique_ages[0]  
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': selected_age})  
heartdisease_probs = q.values
heartdisease_states = q.state_names['heartdisease']  

for state, prob in zip(heartdisease_states, heartdisease_probs):
    print(f"P(heartdisease={state} | age={selected_age}) = {prob}")


print('\n 2. Probability of HeartDisease given cholesterol=100')
unique_chols = heartDisease['chol'].unique()


selected_chol = unique_chols[0]  
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': selected_chol})
heartdisease_probs = q.values
heartdisease_states = q.state_names['heartdisease']  

for state, prob in zip(heartdisease_states, heartdisease_probs):
    print(f"P(heartdisease={state} | chol={selected_chol}) = {prob}")
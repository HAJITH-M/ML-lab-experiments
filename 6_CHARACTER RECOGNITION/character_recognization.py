#pip install -r requirements.txt
import numpy as np
import pandas as pd

data = pd.read_csv('./HR_comma_sep.csv')
data.columns = data.columns.str.strip()
print(data.columns)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data['Departments'] = le.fit_transform(data['Departments'])  # Adjusted column name
X = data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'Departments', 'salary']]
y = data['left']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(6, 5),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01)

clf.fit(X_train, y_train)
ypred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, ypred)
print("Accuracy:", accuracy)

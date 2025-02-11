import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# load the dataset
data = pd.read_csv('Covid.csv')

# convert the Yes/No values to binary
data = data.applymap(lambda x: 1 if x == 'Yes' else 0)

# define the features as all columns except for the COVID-19 column
# define the target as the COVID-19 columnn
X = data.drop(columns=['COVID-19'])
y = data['COVID-19']


# set up the cross validation to have 5 folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# define the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# perform the cross validation
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

# print the results
print("Cross-validation accuracy score", scores)
print("Mean accuracy", np.mean(scores))

# train the final model on the the full dataset
model.fit(X,y)

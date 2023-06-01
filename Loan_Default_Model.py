import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import matthews_corrcoef

np.random.seed = 42

train_data = pd.read_csv("lending_train.csv")
test_data = pd.read_csv("lending_topredict.csv")

print(train_data.shape)

ids = train_data["ID"]

del train_data["ID"]

test_ids = test_data["ID"]

del test_data["ID"]

print(train_data)
train_data = train_data[
    ['loan_duration', 'debt_to_income_ratio', 'fico_score_range_high', 'fico_score_range_low', 'requested_amnt',
     'home_ownership_status', 'annual_income',
     'employment_verified', 'total_revolving_limit', 'fico_inquired_last_6mths', 'loan_paid']]
test_data = test_data[
    ['loan_duration', 'debt_to_income_ratio', 'fico_score_range_high', 'fico_score_range_low', 'requested_amnt',
     'home_ownership_status', 'annual_income',
     'employment_verified', 'total_revolving_limit', 'fico_inquired_last_6mths', 'loan_paid']]

catcols = list(train_data.select_dtypes(include=['object']).astype("category").columns)
numcols = list(train_data.select_dtypes(include=['float64']).columns)

import pandas as pd

for n in numcols:
    train_data[n].fillna(train_data[n].mean(), inplace=True)
    test_data[n].fillna(test_data[n].mean(), inplace=True)

X = train_data
X_test = test_data
Y = X["loan_paid"]
Y_test = X_test["loan_paid"]
X = X.drop("loan_paid", axis=1)
X_test = X_test.drop("loan_paid", axis=1)

undersample = RandomUnderSampler(random_state=42)
X, Y = undersample.fit_resample(X, Y)

print("undersampled")

for c in catcols:
    X = pd.concat([X, pd.get_dummies(X[c], prefix=c, dummy_na=True)], axis=1).drop([c], axis=1)
    X_test = pd.concat([X_test, pd.get_dummies(X_test[c], prefix=c, dummy_na=True)], axis=1).drop([c], axis=1)

print("encoded")

scaler = StandardScaler()
X[numcols] = scaler.fit_transform(X[numcols])
X_test[numcols] = scaler.fit_transform(X_test[numcols])

print("standardized")

X_x_train, X_x__test, y_y_train, y_y_test = train_test_split(X, Y, test_size=.20, random_state=42)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth=10, n_estimators=100, min_samples_leaf=50, max_features='auto', random_state=42)
# fit the predictor and target
rfc.fit(X_x_train, y_y_train)
rfc_predict = rfc.predict(X_x__test)  # check performance
print(rfc_predict)
print('ROCAUC score:', roc_auc_score(y_y_test, rfc_predict))
print('Accuracy score:', accuracy_score(y_y_test, rfc_predict))
print('F1 score:', f1_score(y_y_test, rfc_predict))
print(precision_recall_fscore_support(y_y_test, rfc_predict, average='macro'))
print("MCC: " + str(matthews_corrcoef(y_y_test, rfc_predict)))

final = pd.DataFrame()

final["ID"] = test_ids
final["loan_paid"] = rfc.predict(X_test)
print(final)
final.to_csv("Loan_ToSubmit.csv", index=False)

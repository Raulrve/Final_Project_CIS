import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn
import missingno
import matplotlib_inline

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

from sklearn.metrics import accuracy_score


def main():
    train_df = pd.read_csv('Raw Data/train.csv')
    test_df = pd.read_csv('Raw Data/test.csv')
    ans_df = pd.read_csv('Raw Data/gender_submission.csv')
    ans_df = ans_df.drop(['PassengerId'], axis=1)

    imp_age = IterativeImputer(max_iter=150, random_state=34, n_nearest_features=3)
    imp_fare = IterativeImputer(max_iter=150, random_state=34, n_nearest_features=3)
    imp_embarked = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

    target_df = train_df['Survived']

    # Impute Age on both dataframes
    train_df["Age"] = np.round(imp_age.fit_transform(train_df[["Age"]]))
    test_df["Age"] = np.round(imp_age.fit_transform(test_df[["Age"]]))

    # Impute the Embarked on both dataframes
    train_df["Embarked"] = imp_embarked.fit_transform(train_df[["Embarked"]])
    test_df["Embarked"] = imp_embarked.fit_transform(test_df[["Embarked"]])

    # Impute the Embarked on both dataframes
    train_df["Fare"] = imp_fare.fit_transform(train_df[["Fare"]])
    test_df["Fare"] = imp_fare.fit_transform(test_df[["Fare"]])

    # Create the Family Size Feature
    train_df["FamSize"] = train_df["SibSp"] + train_df["Parch"]
    test_df["FamSize"] = test_df["SibSp"] + test_df["Parch"]

    # Titles
    train_df["Title"] = train_df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    test_df["Title"] = test_df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

    combined_df = train_df.append(test_df)
    combined_df.reset_index(inplace=True)

    combined_df = pd.get_dummies(columns=["Pclass", "Sex", "Embarked", "FamSize", "Title"], data=combined_df, drop_first=False)

    train_df = combined_df.iloc[:891]
    test_df = combined_df.iloc[891:]

    train_df = train_df.drop(['Name', 'Survived', 'PassengerId', 'index', 'Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Name', 'Survived', 'PassengerId', 'index', 'Ticket', 'Cabin'], axis=1)

    logreg = LogisticRegression(max_iter=10000)
    randomForest = RandomForestClassifier()
    kneighbors = KNeighborsClassifier()
    decisionTree = tree.DecisionTreeClassifier()

    models = [logreg, randomForest, kneighbors, decisionTree]

    ans_df = pd.read_csv('Raw Data/gender_submission.csv')
    ans_df = ans_df.drop(['PassengerId'], axis=1)

    ans_numpy_arr = ans_df.values
    ans_numpy_arr = ans_numpy_arr.flatten()

    def getAccuracy(models, target_df, test_df, ans_numpy_arr):
        for x in models:
            x.fit(train_df, target_df)
            predict = x.predict(test_df)
            ans = accuracy_score(ans_numpy_arr, predict)

            output = x.predict(test_df).astype(int)
            df_output = pd.DataFrame()
            aux = pd.read_csv('./Raw Data/test.csv')
            df_output['PassengerId'] = aux['PassengerId']
            df_output['Survived'] = output

            theModelStr = str(x)
            theModelStr = theModelStr.split('(')[0]

            df_output[['PassengerId', 'Survived']].to_csv('./Results/' + str(theModelStr) + '.csv', index=False)

            print('Accuracy of : {0}'.format(x.__class__))
            print(ans, "\n")

    getAccuracy(models, target_df, test_df, ans_numpy_arr)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
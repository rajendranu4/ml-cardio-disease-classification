import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

import logging
logger = logging.getLogger('log')


def setup_logging(logger_name, filename):
    logger.info("*****Invoked setup_logging()*****")
    logging.basicConfig(filename="std.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    logger.info("*****Exiting setup_logging()*****")
    return log


# checking correlation between features
def correlation_matrix(df):
    logger.info("*****Invoked correlation_matrix()*****")
    corrMatrix = round(df.corr(), 2)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(corrMatrix, annot=True, ax=ax)
    plt.show()
    logger.info("*****Exiting correlation_matrix()*****")


def get_model(model_name):
    logger.info("*****Invoked get_models()*****")

    if model_name.lower() == 'decision tree':
        model = DecisionTreeClassifier()

    elif model_name.lower() == 'knn':
        model = KNeighborsClassifier(weights='distance')

    elif model_name.lower() == 'naive bayes':
        model = GaussianNB()

    elif model_name.lower() == 'random forest':
        model = RandomForestClassifier(random_state=0)

    elif model_name.lower() == 'sgd':
        model = SGDClassifier(loss='hinge')

    else:
        logger.info("Selected model is not in the list of models. "
                    "Please select appropriate models from the given list")

    logger.info("*****Exiting get_models()*****")
    return model


def data_preprocessing(df):
    logger.info("*****Invoked data_preprocessing()*****")
    print("\nDataframe")
    print(df.head())
    print("\nDataframe summary")
    print(df.info())

    logger.info("Shape of the dataset: {}".format(df.shape))
    # checking count of instances in each class
    print("\nChecking count of instances in each class")
    print(df['cardio'].value_counts())
    logger.info("Count of instances in each class")
    logger.info(df['cardio'].value_counts())

    # checking for missing values
    print("\nChecking for null values")
    print(df.isna().sum())

    correlation_matrix(df)

    df = df.iloc[:, 2:]
    print("\nRemoved ID column and age_days column - age_days and age_years have correlation value of 1")
    print(df.head())
    logger.info("Removed ID column and age_days column as age_days and age_years have correlation value of 1")

    X = df.iloc[:, :-1]
    y = df['cardio']

    logger.info("*****Exiting data_preprocessing()*****")
    return X, y
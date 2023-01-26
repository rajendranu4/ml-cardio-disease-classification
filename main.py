import pandas as pd

import helper_funcs as helpers
from Classifier import Classifier


if __name__ == '__main__':
    logger = helpers.setup_logging('log', 'out.log')

    logger.info("*****Reading the dataset")
    # reading the dataset
    df_cardio = pd.read_csv("data/data.csv")
    models_list = [
        {
            'model_name': 'Logistic Regression',
            'model_params': {'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],'max_iter':range(80,140)}
        },
        {
            'model_name': 'Decision Tree',
            'model_params': {'criterion':['gini', 'entropy']}
        },
        {
            'model_name': 'KNN',
            'model_params': {'n_neighbors':range(3,8)}
        },
        {
            'model_name': 'Naive Bayes',
            'model_params': {}
        },
        {
            'model_name': 'Random Forest',
            'model_params': {'n_estimators':range (90,210,20)}
        },
        {
            'model_name': 'SGD',
            'model_params': {}
        }
    ]

    data_X, data_y = helpers.data_preprocessing(df_cardio)

    for model in models_list:
        logger.info("############Starting {}############".format(model['model_name']))
        classifier = Classifier(model['model_name'], model['model_params'])
        classifier.cv_train_test(n_fold=5, X=data_X, y=data_y)
        classifier.cv_evaluation_metrics(n_fold=5)
        classifier.auc_roc()


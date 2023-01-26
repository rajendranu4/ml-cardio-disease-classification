import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import helper_funcs as helpers

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import logging
logger = logging.getLogger('log')


class Classifier:

    def __init__(self, name, grid_search_params):
        logger.info("*****Instantiating Classifier*****")
        logger.info("Name: {}".format(name))
        logger.info("Grid Search Params:")
        logger.info(grid_search_params)
        self.name = name
        self.grid_search_params = grid_search_params
        self.model = helpers.get_model(self.name)

        # dictionary to keep track of accuracy of all the models in each iteration of kfolds
        # sample for 3fold CV: model_accuracies = {'Logistic_Regression':[0.7,0.8,0.7], 'KNN':[0.8,0.9,0.8]}
        # dictionary to keep track of parameters for plotting roc_auc of all the models from each iteration of kfolds
        # sample for 3fold CV: model_auc_roc = {'Logistic_Regression':[(fpr,tpr,auc), (fpr,tpr,auc),..], 'KNN':[(fpr,tpr,auc),(fpr,tpr,auc),.]}
        # fpr --> false positive rates, tpr --> true positive rates, auc --> auc score
        self.model_accuracies = []
        self.model_metrics = []
        self.model_auc_roc = []
        self.model_features = []

    def get_features(self, data_X, data_y, select_n_features):
        logger.info("*****Invoking get_features()*****")
        # SFBS using KNN as classifier
        # knn = KNeighborsClassifier(n_neighbors=3)
        sfbs = SFS(self.model,
                   k_features=select_n_features,  # to select number of features
                   forward=False,  # to select forward/backward
                   floating=True,  # to set if floating required or not
                   scoring='accuracy',
                   cv=10,  # 10 fold cross validation
                   n_jobs=-1
                   )

        # after applying sfbs fit the data:
        # features = ['age_year', 'gender', 'height', 'weight', 'ap_hi',
        #             'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

        sfbs.fit(data_X, data_y)
        logger.info("*****Exiting get_features()*****")
        return sfbs

    def cv_train_test(self, n_fold, X, y):
        logger.info("*****Invoking cv_train_test()*****")
        # print("\nCheck point 1....")
        print("\n Stratified KFold with {} folds".format(n_fold))

        # create n stratified folds for CV
        s_kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=1)

        curr_iter = 1

        logger.info("Starting {}-fold stratified CV".format(n_fold))

        # tqdm is just for visualizing the progress
        # parameter y is enough to split the indices in kfold stratified CV, np.zeros(len(y)) is just placeholder for X
        # split() returns the training instance indices and testing instance indices corresponding to the nfolds
        for train_index, test_index in tqdm(s_kfold.split(np.zeros(len(y)), y), total=s_kfold.get_n_splits()):
            logger.info("Iteration {}".format(curr_iter))
            print("\nIteration {}".format(curr_iter))
            iter_start_time = time.time()
            # print("Start time: {}".format(iter_start_time))

            # print("\ncheck point 2....")
            # separating X and y for training and testing
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # scaling training and testing dataset separately
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # each iteration of k-fold stratified CV applied for all the models
            print("Running model: {}".format(self.model))
            model_start_time = time.time()
            # print("Start time: {}".format(model_start_time))
            # print("\ncheck point 3....")

            # selecting features with the training dataset
            logger.info("Finding Feature subset")
            print("Finding Feature subset....")
            features_subset = self.get_features(data_X=X_train, data_y=y_train, select_n_features=6)

            fs_current = list(features_subset.k_feature_names_)
            '''print("Model: {}".format(model))
            print("Current Features: {}".format(fs_current))
            print("Overall features without current features: {}".format(model_features[model]))'''

            # to store the features selected at each iteration
            additional_diff = np.setdiff1d(fs_current, self.model_features)
            self.model_features.extend(additional_diff)

            # print("Overall features with current features: {}".format(model_features[model]))

            # transforming the training dataset with the features selected from above step
            X_train_subset = features_subset.transform(X_train)

            # transforming the testing dataset with the features selected from above step
            X_test_subset = features_subset.transform(X_test)

            # finding best hyperparameters for
            logger.info("Tuning Hyperparameters")
            print("Tuning hyperparameters....")
            grid_search = GridSearchCV(self.model, self.grid_search_params, cv=10, scoring='accuracy')
            grid_search.fit(X_train_subset, y_train)
            # print(grid_search.best_estimator_)
            current_model = grid_search.best_estimator_

            logger.info("Training model starts here")
            print("Training model....")
            current_model.fit(X_train_subset, y_train)

            # since SGD with hinge loss does not have probability estimates, calibrating using below for probability estimates
            if self.model == 'SGD':
                # cv=prefit tells calibrator that the current_model is already fit with training data
                # calibrated model is again taken as current_model and proceeded further
                calibrated_classfier = CalibratedClassifierCV(current_model, cv='prefit')
                current_model = calibrated_classfier.fit(X_train_subset, y_train)

            # accuracy score of a model is calculated using the training dataset and
            # it is appended to dictionary for corresponding iteration of kfold

            logger.info("Evaluating trained model")
            print("Evaluating model....")
            # model_accuracies[model].append(current_model.score(X_test_subset, y_test))
            pred_y = current_model.predict(X_test_subset)

            self.model_accuracies.append(accuracy_score(y_test, pred_y))
            self.model_metrics.append(precision_recall_fscore_support(y_test, pred_y, average='binary'))

            # roc_auc works on probabilities, hence predicting probabilities using predict_proba() with testing dataset
            proba_ = current_model.predict_proba(X_test_subset)

            # keeping only the positive classes from the probabilities (proba_[:, 1]) and
            # calculated false positive rates and true positive rates
            fpr, tpr, _ = roc_curve(y_test, proba_[:, 1])
            roc_auc = auc(fpr, tpr)

            # fpr, tpr and auc score is recorded for this model for this particular iteration of kfolds
            self.model_auc_roc.append((fpr, tpr, roc_auc))
            model_end_time = time.time() - model_start_time
            logger.info("Model total time: {}".format(model_end_time))
            print("Model total time: {}".format(model_end_time))

            iter_end_time = time.time() - iter_start_time
            logger.info("Iteration {} total time: {}".format(curr_iter, iter_end_time))
            print("Iteration {} total time: {}".format(curr_iter, iter_end_time))
            curr_iter += 1

            print("\nThe below features are selected by the model atleast "
                  "once during the feature selection phase in {} fold CV".format(n_fold))
            logger.info("Selected Features by the model")
            logger.info("Features: {}".format(self.model_features))
            print("Features: {}".format(self.model_features))

    def cv_evaluation_metrics(self, n_fold):
        logger.info("*****Invoking cv_evaluation_metrics()*****")
        precision = 0
        recall = 0
        fscore = 0
        for iter_each in self.model_metrics:
            precision += iter_each[0]
            recall += iter_each[1]
            fscore += iter_each[2]
        precision = precision / n_fold
        recall = recall / n_fold
        fscore = fscore / n_fold
        print("\nModel: {}".format(self.model))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("FScore: {}".format(fscore))
        print("Accuracy: {}".format(np.mean(self.model_accuracies)))
        logger.info("Precision: {}".format(precision))
        logger.info("Precision: {}".format(precision))
        logger.info("Recall: {}".format(recall))
        logger.info("FScore: {}".format(fscore))
        logger.info("Accuracy: {}".format(np.mean(self.model_accuracies)))
        logger.info("*****Exiting cv_evaluation_metrics()*****")

    def auc_roc(self):
        logger.info("*****Invoking auc_roc()*****")
        # plotting graph for all models with the roc_auc parameters from model_auc_roc dictionary
        plt.figure(figsize=(7, 7))
        for i, score in enumerate(self.model_auc_roc):
            fpr = score[0]
            tpr = score[1]
            roc_auc = score[2]
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC for fold %d (AUC = %0.2f)' % (i + 1, roc_auc))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Cross-Validation ROC: ' + self.name, fontsize=15)
        plt.legend()
        plt.show()
        print("\n")

    logger.info("*****Exiting auc_roc()*****")
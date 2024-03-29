# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u

# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        answer = u.starter_code()
        return answer

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

#     def partB(
#         self,
#     ):
#         X, y, Xtest, ytest = u.prepare_data()
#         Xtrain, ytrain = u.filter_out_7_9s(X, y)
#         Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
#         Xtrain = nu.scale_data(Xtrain)
#         Xtest = nu.scale_data(Xtest)

#         answer = {}

#         # Enter your code and fill the `answer` dictionary

#         answer["length_Xtrain"] = None  # Number of samples
#         answer["length_Xtest"] = None
#         answer["length_ytrain"] = None
#         answer["length_ytest"] = None
#         answer["max_Xtrain"] = None
#         answer["max_Xtest"] = None
#         return answer, Xtrain, ytrain, Xtest, ytest
    
    def partB(
        self,
    ):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)

        answer = {}

        # Enter your code and fill the `answer` dictionary

        answer["length_Xtrain"] = len(Xtrain)  # Number of samples
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = np.max(Xtrain)
        answer["max_Xtest"] = np.max(Xtest)
        return answer, Xtrain, ytrain, Xtest, ytest


    


    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
#     def partC(
#         self,
#         X: NDArray[np.floating],
#         y: NDArray[np.int32],
#     ):
#         # Enter your code and fill the `answer` dictionary

#         answer = {}
#         answer["clf"] = None  # the estimator (classifier instance)
#         answer["cv"] = None  # the cross validator instance
#         # the dictionary with the scores  (a dictionary with
#         # keys: 'mean_fit_time', 'std_fit_time', 'mean_accuracy', 'std_accuracy'.
#         answer["scores"] = None
#         return answer
    
    def partC(self, X, y):
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        scores = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)

        answer = {
            "clf": clf,
            "cv": cv,
            "scores": {
                "mean_fit_time": scores['fit_time'].mean(),
                "std_fit_time": scores['fit_time'].std(),
                "mean_accuracy": scores['test_score'].mean(),
                "std_accuracy": scores['test_score'].std()
            }
        }
        return answer


    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

#     def partD(
#         self,
#         X: NDArray[np.floating],
#         y: NDArray[np.int32],
#     ):
#         # Enter your code and fill the `answer` dictionary

#         # Answer: same structure as partC, except for the key 'explain_kfold_vs_shuffle_split'

#         answer = {}
#         answer["clf"] = None
#         answer["cv"] = None
#         answer["scores"] = None
#         answer["explain_kfold_vs_shuffle_split"] = None
    
    # Part D of Section1 in part_1_template_solution.py

    def partD(self, X, y):
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.seed)
        scores = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)

        explanation = """
        Pros of Shuffle-Split:
        - Allows for a more flexible choice of the number of iterations and the size of the training and test sets.
        - Useful for large datasets or for quick model evaluations.

        Cons of Shuffle-Split:
        - May introduce more variability in the performance estimates compared to k-fold CV.
        - Less systematic coverage of all data points compared to k-fold CV.
        """

        answer = {
            "clf": clf,
            "cv": cv,
            "scores": {
                "mean_fit_time": scores['fit_time'].mean(),
                "std_fit_time": scores['fit_time'].std(),
                "mean_accuracy": scores['test_score'].mean(),
                "std_accuracy": scores['test_score'].std()
            },
            "explain_kfold_vs_shuffle_split": explanation.strip()
        }

        return answer


    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

#     def partE(
#         self,
#         X: NDArray[np.floating],
#         y: NDArray[np.int32],
#     ):
#         # Answer: built on the structure of partC
#         # `answer` is a dictionary with keys set to each split, in this case: 2, 5, 8, 16
#         # Therefore, `answer[k]` is a dictionary with keys: 'scores', 'cv', 'clf`

#         answer = {}

#         # Enter your code, construct the `answer` dictionary, and return it.

#         return answer
    
    # Part E of Section1 in part_1_template_solution.py

    def partE(self, X, y):
        answer = {}
        k_values = [2, 5, 8, 16]

        for k in k_values:
            cv = ShuffleSplit(n_splits=k, test_size=0.2, random_state=self.seed)
            scores = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=DecisionTreeClassifier(random_state=self.seed), cv=cv)

            answer[k] = {
                "scores": {
                    "mean_accuracy": scores['test_score'].mean(),
                    "std_accuracy": scores['test_score'].std()
                },
                "cv": cv,
                "clf": DecisionTreeClassifier(random_state=self.seed)
            }

        return answer


    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

#     def partF(
#         self,
#         X: NDArray[np.floating],
#         y: NDArray[np.int32],
#     ) -> dict[str, Any]:
#         """ """

#         answer = {}

#         # Enter your code, construct the `answer` dictionary, and return it.

#         """
#          Answer is a dictionary with the following keys: 
#             "clf_RF",  # Random Forest class instance
#             "clf_DT",  # Decision Tree class instance
#             "cv",  # Cross validator class instance
#             "scores_RF",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
#             "scores_DT",  Dictionary with keys: "mean_fit_time", "std_fit_time", "mean_accuracy", "std_accuracy"
#             "model_highest_accuracy" (string)
#             "model_lowest_variance" (float)
#             "model_fastest" (float)
#         """

#         return answer
    
    # Part F of Section1 in part_1_template_solution.py

    def partF(self, X, y):
        clf_RF = RandomForestClassifier(random_state=self.seed)
        clf_DT = DecisionTreeClassifier(random_state=self.seed)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.seed)

        scores_RF = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf_RF, cv=cv)
        scores_DT = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf_DT, cv=cv)

        answer = {
            "clf_RF": clf_RF,
            "clf_DT": clf_DT,
            "cv": cv,
            "scores_RF": {
                "mean_accuracy": scores_RF['test_score'].mean(),
                "std_accuracy": scores_RF['test_score'].std()
            },
            "scores_DT": {
                "mean_accuracy": scores_DT['test_score'].mean(),
                "std_accuracy": scores_DT['test_score'].std()
            },
            # Determine which model has higher average accuracy and lower variance
            "model_highest_accuracy": "RF" if scores_RF['test_score'].mean() > scores_DT['test_score'].mean() else "DT",
            "model_lowest_variance": "RF" if scores_RF['test_score'].std() < scores_DT['test_score'].std() else "DT",
            "model_fastest": "RF" if scores_RF['fit_time'].mean() < scores_DT['fit_time'].mean() else "DT"
        }

        return answer


    # ----------------------------------------------------------------------
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """
    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - clf: The classifier instance to use for classification.
        - cv: The cross validator instance to use for cross validation.
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """

        # refit=True: fit with the best parameters when complete
        # A test should look at best_index_, best_score_ and best_params_
        """
        List of parameters you are allowed to vary. Choose among them.
         1) criterion,
         2) max_depth,
         3) min_samples_split, 
         4) min_samples_leaf,
         5) max_features 
         5) n_estimators
        """
        param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        #"n_estimators":[50,100,200]
        }
        cv = ShuffleSplit(n_splits=5, random_state=self.seed)
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X,y)
        # Predictions with the initial model
        y_train_pred_orig = rf.predict(X)
        y_test_pred_orig = rf.predict(Xtest)

        # Confusion matrices
        conf_matrix_train_orig = confusion_matrix(y, y_train_pred_orig)
        conf_matrix_test_orig = confusion_matrix(ytest, y_test_pred_orig)

        # Accuracies
        accuracy_train_orig = nu.accuracy(conf_matrix_train_orig)#accuracy_score(y, y_train_pred_orig)
        accuracy_test_orig = nu.accuracy(conf_matrix_test_orig)#accuracy_score(ytest, y_test_pred_orig)

# Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='accuracy')

        # Perform grid search
        grid_search.fit(X, y)
        best_clf = grid_search.best_estimator_
        accuracy = best_clf.score(Xtest,ytest)
        # print('%'*20)
        # print(best_clf)
        # Predictions with the optimized model
        # y_train_pred_best = best_clf.predict(X)
        # y_test_pred_best = best_clf.predict(Xtest)

        mean_test_scores = grid_search.cv_results_['mean_test_score']
        # Calculate the mean accuracy
        mean_accuracy = mean_test_scores.mean()
        best_rf_clf = best_clf
        #best_rf_clf.fit(X,y)
        y_train_pred_best = best_rf_clf.predict(X)
        y_test_pred_best = best_rf_clf.predict(Xtest)

        # y_train_pred_best = best_clf.predict(X)
        # y_test_pred_best = best_clf.predict(Xtest)

        # Confusion matrices
        conf_matrix_train_best = confusion_matrix(y, y_train_pred_best)
        conf_matrix_test_best = confusion_matrix(ytest, y_test_pred_best)

        # Accuracies
        accuracy_train_best = nu.accuracy(conf_matrix_train_best) #accuracy_score(y, y_train_pred_best)
        accuracy_test_best = nu.accuracy(conf_matrix_test_best) #accuracy_score(ytest, y_test_pred_best)


        answer = {
    "clf": rf,
    "default_parameters": rf.get_params(),
    "best_estimator": best_clf,
    "grid_search": grid_search,
    "mean_accuracy_cv": mean_accuracy,
    "confusion_matrix_train_orig": conf_matrix_train_orig,
    "confusion_matrix_train_best": conf_matrix_train_best,
    "confusion_matrix_test_orig": conf_matrix_test_orig,
    "confusion_matrix_test_best": conf_matrix_test_best,
    "accuracy_orig_full_training": accuracy_train_orig,
    "accuracy_best_full_training": accuracy_train_best,
    "accuracy_orig_full_testing": accuracy_test_orig,
    "accuracy_best_full_testing": accuracy_test_best,
}

# Now, you can print or return the answer dictionary.


        # Enter your code, construct the `answer` dictionary, and return it.

        """
           `answer`` is a dictionary with the following keys: 
            
            "clf", base estimator (classifier model) class instance
            "default_parameters",  dictionary with default parameters 
                                   of the base estimator
            "best_estimator",  classifier class instance with the best
                               parameters (read documentation)
            "grid_search",  class instance of GridSearchCV, 
                            used for hyperparameter search
            "mean_accuracy_cv",  mean accuracy score from cross 
                                 validation (which is used by GridSearchCV)
            "confusion_matrix_train_orig", confusion matrix of training 
                                           data with initial estimator 
                                (rows: true values, cols: predicted values)
            "confusion_matrix_train_best", confusion matrix of training data 
                                           with best estimator
            "confusion_matrix_test_orig", confusion matrix of test data
                                          with initial estimator
            "confusion_matrix_test_best", confusion matrix of test data
                                            with best estimator
            "accuracy_orig_full_training", accuracy computed from `confusion_matrix_train_orig'
            "accuracy_best_full_training"
            "accuracy_orig_full_testing"
            "accuracy_best_full_testing"
               
        """
        # The mean accuracy of Cross validation is around 65% where as when the model trained on the enitre set, It has an 100% acccuracy, So it is higher than that of the mean accuracy of CV.
        return answer

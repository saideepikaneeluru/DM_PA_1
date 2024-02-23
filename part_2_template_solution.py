# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, ShuffleSplit
# Any other imports you need


# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
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
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

#     def partA(
#         self,
#     ) -> tuple[
#         dict[str, Any],
#         NDArray[np.floating],
#         NDArray[np.int32],
#         NDArray[np.floating],
#         NDArray[np.int32],
#     ]:
#         answer = {}
#         # Enter your code and fill the `answer`` dictionary

#         # `answer` is a dictionary with the following keys:
#         # - nb_classes_train: number of classes in the training set
#         # - nb_classes_test: number of classes in the testing set
#         # - class_count_train: number of elements in each class in the training set
#         # - class_count_test: number of elements in each class in the testing set
#         # - length_Xtrain: number of elements in the training set
#         # - length_Xtest: number of elements in the testing set
#         # - length_ytrain: number of labels in the training set
#         # - length_ytest: number of labels in the testing set
#         # - max_Xtrain: maximum value in the training set
#         # - max_Xtest: maximum value in the testing set

#         # return values:
#         # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

#         Xtrain = Xtest = np.zeros([1, 1], dtype="float")
#         ytrain = ytest = np.zeros([1], dtype="int")

#         return answer, Xtrain, ytrain, Xtest, ytest
    
    
    def partA(self):
        # Load the full MNIST dataset
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X = X.astype(np.float64)
        y = y.astype(np.int32)

        # Normalize the data if required
        if self.normalize:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

        # Split the data into training and testing sets
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=1-self.frac_train, random_state=self.seed)

        # Analyze class distribution
        classes, class_count_train = np.unique(ytrain, return_counts=True)
        _, class_count_test = np.unique(ytest, return_counts=True)
        nb_classes_train = len(classes)
        nb_classes_test = len(classes)  # Should be the same as nb_classes_train

        # Prepare the answer dictionary
        answer = {
            'nb_classes_train': nb_classes_train,
            'nb_classes_test': nb_classes_test,
            'class_count_train': class_count_train.tolist(),
            'class_count_test': class_count_test.tolist(),
            'length_Xtrain': len(Xtrain),
            'length_Xtest': len(Xtest),
            'length_ytrain': len(ytrain),
            'length_ytest': len(ytest),
            'max_Xtrain': np.max(Xtrain),
            'max_Xtest': np.max(Xtest),
        }

        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

#     def partB(
#         self,
#         X: NDArray[np.floating],
#         y: NDArray[np.int32],
#         Xtest: NDArray[np.floating],
#         ytest: NDArray[np.int32],
#         ntrain_list: list[int] = [],
#         ntest_list: list[int] = [],
#     ) -> dict[int, dict[str, Any]]:
#         """ """
#         # Enter your code and fill the `answer`` dictionary
#         answer = {}

#         """
#         `answer` is a dictionary with the following keys:
#            - 1000, 5000, 10000: each key is the number of training samples

#            answer[k] is itself a dictionary with the following keys
#             - "partC": dictionary returned by partC section 1
#             - "partD": dictionary returned by partD section 1
#             - "partF": dictionary returned by partF section 1
#             - "ntrain": number of training samples
#             - "ntest": number of test samples
#             - "class_count_train": number of elements in each class in
#                                the training set (a list, not a numpy array)
#             - "class_count_test": number of elements in each class in
#                                the training set (a list, not a numpy array)
#         """

#         return answer
    


    def partB(self, X, y, ntrain_list=[1000, 5000, 10000], ntest_list=[200, 1000, 2000]):
        answer = {}

        for ntrain, ntest in zip(ntrain_list, ntest_list):
            # Slice the dataset for the current ntrain and ntest
            Xtrain = X[:ntrain]
            ytrain = y[:ntrain]
            Xtest = X[ntrain:ntrain+ntest]
            ytest = y[ntrain:ntrain+ntest]

            # Logistic Regression for multi-class classification
            clf = LogisticRegression(max_iter=300, solver='lbfgs', multi_class='multinomial', random_state=self.seed)

            # Fit the model
            clf.fit(Xtrain, ytrain)

            # Cross-validation with ShuffleSplit for training data
            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.seed)
            scores = cross_val_score(clf, Xtrain, ytrain, cv=cv)

            # Training and testing scores
            train_score = clf.score(Xtrain, ytrain)
            test_score = clf.score(Xtest, ytest)

            # Class distribution in training and testing sets
            classes, class_count_train = np.unique(ytrain, return_counts=True)
            _, class_count_test = np.unique(ytest, return_counts=True)

            answer[ntrain] = {
                "ntrain": ntrain,
                "ntest": ntest,
                "class_count_train": class_count_train.tolist(),
                "class_count_test": class_count_test.tolist(),
                "mean_cv_score": np.mean(scores),
                "std_cv_score": np.std(scores),
                "train_score": train_score,
                "test_score": test_score,
            }

        return answer


    
    

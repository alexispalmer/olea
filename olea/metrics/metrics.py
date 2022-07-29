import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import auc, roc_curve
# from distinctipy import distinctipy
import os as os
from matplotlib import pyplot as plt

class Metrics:
    y_true = []
    y_pred = []
    y_true_names = []
    y_pred_names = []
    name_to_idx = {}
    n_classes = 0
    digits = 4
    fpr = None
    tpr = None
    roc_auc = None

    @staticmethod
    def classification_report(y_true, y_pred) -> None:
        """Prints out the classification report to console.
        """
        print(classification_report(y_true, y_pred, digits = 4,zero_division=0))

    @staticmethod   
    def get_metrics_dictionary(y_true, y_pred) -> dict:
        """Returns a dictionary denoting the metrics of the model. 
        It is the accessible version of classification_report()
        
        Returns:
            dict: A dictionary containing the metrics of the model with respect to the classes that it predicts.
        """
        return classification_report(y_true, y_pred, 
                                     digits = 4, 
                                     output_dict = True,
                                     zero_division = 0)

    @classmethod
    def __assert_equal_length(cls) -> None:

        assert (len(cls.y_true) == len(cls.y_pred)), "The length of the true labels is not the same as the pred labels."
    
    @classmethod
    def __check_equal_number_of_classes(cls) -> None:
        if len(set(cls.y_true)) != len(set(cls.y_pred)):
            warnings.warn("WARNING: The number of classes in the true labels and the predicted labels is not the same.")

    @classmethod
    def set_labels(cls, y_true, y_pred) -> None:
        """Set the names of the labels.

        Args:
            y_true (list): A list containing the true labels of the data.
            y_pred (list): A list containing the predicted labels of the data.
        """
        cls.y_true_names = y_true
        cls.y_pred_names = y_pred
        cls.n_classes = len(set(y_true))
        cls.__assert_equal_length()
        cls.__check_equal_number_of_classes()

    @classmethod
    def get_name_to_idx_dict(cls, y_true, y_pred) -> dict:
        cls.set_labels(y_true, y_pred)
        cls.y_true_names = list(set(cls.y_true))
        cls.y_true_names.sort()
        cls.y_pred_names = list(set(cls.y_pred))
        cls.y_pred_names.sort()
        cls.name_to_idx = {}
        cls.name_to_idx = dict(enumerate(cls.y_true_names))
        cls.name_to_idx = dict([(value, key) for key, value in 
                                 cls.name_to_idx.items()])
        return cls.name_to_idx
    
    @classmethod
    def __prettify_confusion_matrix(cls, confusion_matrix) -> str:
        """Prints out the confusion matrix with labels to console.
        Args:
            confusion_matrix: Uses the confusion matrix to format it in a pretty format. 
        Returns:
            str: Returns a pretty printed confusion matrix.
        """
        pandas_cf = pd.DataFrame(confusion_matrix,
                                 index = cls.y_true_names,
                                 columns = cls.y_true_names).round(cls.digits)
        print(pandas_cf)

    @classmethod
    def confusion_matrix(cls, y_true, y_pred) -> None:
        """Prints out the confusion matrix to console.
        """
        cls.set_labels(y_true, y_pred)
        cls.__prettify_confusion_matrix(confusion_matrix(cls.y_true, cls.y_pred))
    
    @classmethod
    def norm_confusion_matrix(cls, along = 'row') -> None:
        """Prints out a normalized confusion matrix to console, either normalized along the rows or columns.

        Args:
            along (str, optional): Takes 'row' or 'col' as arguments. Defaults to 'row'.
            The user specifies if the confusion matrix should be normalized along its rows or columns. 
        """
        AXIS_NUM = 1 if along == 'row' else 0

        DENOMINATOR = (cls.__confusion_matrix.sum(axis = AXIS_NUM)[:, np.newaxis])
        if AXIS_NUM:
            NUMERATOR = cls.__confusion_matrix.astype('float')
            norm_conf_mat = (NUMERATOR / DENOMINATOR)
        else:
            NUMERATOR = cls.__confusion_matrix.T.astype('float')
            norm_conf_mat = (NUMERATOR / DENOMINATOR).T
        rounded_matrix = np.around(norm_conf_mat, decimals = cls.digits)
        cls.__prettify_confusion_matrix(rounded_matrix)

    @classmethod
    def _setup_aucroc(cls, y_softmax_probs: np.array):
        """A protected method to setup the auc-roc plot.

        Args:
            y_softmax_probs (np.array): This is an np.array of the predicted probabilities of the data. The np.array that contains the softmax probs of all classes of all predictions.
        """
        # Much code for this function is adapted from
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_true_as_idx = np.array([cls.name_to_idx.get(x) for x in cls.y_true])
        for i in range(cls.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_as_idx == i, y_softmax_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(cls.n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(cls.n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= cls.n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        cls.fpr = fpr
        cls.tpr = tpr
        cls.roc_auc = roc_auc  

    @staticmethod
    def plot_roc_curve(cls, y_softmax_probs: np.array, save = True,
                       image_filepath = os.getcwd(), 
                       image_filename = "roc_curve", 
                       class_line_width = 1,
                       macro_average_line_width = 3,
                       color_list = []) -> None:
        """This plots the roc curve in a new window and saves it to file if the 
        user so chooses. 

        Args:
            y_softmax_probs (np.array): This is an np.array of the predicted probabilities of the data. The np.array that contains the softmax probs of all classes of all predictions.
            save (bool, optional): Save the generated image to file? Defaults to True.
            image_filepath (str, optional): The path to which the image is saved. Defaults to os.getcwd().
            image_filename (str, optional): The name of the image file. Defaults to "roc_curve".
            class_line_width (int, optional): Controls how wide the lines the multiple classes should be. Defaults to 1.
            macro_average_line_width (int, optional): Controls how wide the dashed line of the macro average of the ROC curves is. Defaults to 3.
            color_list (list, optional): A list of colors. Length should be the same as the number of classes. Defaults to []. If the user doesn't specify a color list, self._distinct_colors will be used, which is automatically generated.
        """
        cls._setup_aucroc(y_softmax_probs)
        # Much code for this function is adapted from
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        class_line_width = class_line_width
        macro_average_line_width = macro_average_line_width
        plt.figure()
        plt.plot(
            cls.fpr["macro"],
            cls.tpr["macro"],
            label = "Macro-Avg, (AUC = {0:0.3f})".format(cls.roc_auc["macro"]),
            color = "navy",
            linestyle = ":",
            linewidth = macro_average_line_width)
        for i, color in zip(range(len(cls.y_true)), color_list):
            plt.plot(
                cls.fpr[i],
                cls.tpr[i],
                color = color,
                lw = class_line_width,
                label = "{0}, (AUC = {1:0.3f})".format(cls.y_true_names[i], cls.roc_auc[i]))
        plt.plot([0, 1], [0, 1], "k--", lw = class_line_width)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for All Classes")
        plt.legend(loc="center left", 
            bbox_to_anchor=(1, 0.5), 
            ncol = 1, 
            fancybox = True, 
            shadow = True)
        if save:
            plt.savefig(image_filepath + "/" + image_filename + ".png", 
                        bbox_inches='tight')
        plt.show()
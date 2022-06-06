import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import auc, roc_curve
from distinctipy import distinctipy
import os as os
from matplotlib import pyplot as plt

class Metrics:
    def __init__(self, y_true: list, y_pred: list) -> None:
        """Initialization of the Metrics class.

        Args:
            y_true (list): A list containing the true labels of the data.
            y_pred (list): A list containing the predicted labels of the data.
        """
        self.__assert_equal_length(y_true, y_pred)
        self.__check_equal_number_of_classes(y_true, y_pred)
        self.y_true = y_true
        self.y_pred = y_pred
        self.n_classes = len(set(y_true))
        self.y_true_names = list(set(y_true))
        self.y_true_names.sort()
        self.y_pred_names = list(set(y_pred))
        self.y_pred_names.sort()
        self.name_to_idx = {}
        self.__generate_name_to_idx_dict()
        self._distinct_colors = distinctipy.get_colors(n_colors = self.n_classes)
        self._distinct_colors = [distinctipy.get_hex(x) for x in self._distinct_colors]
        self._fpr = None
        self._tpr = None
        self._roc_auc = None
        self._digits = 4
        self.__confusion_matrix = confusion_matrix(y_true, y_pred)
        self.__classification_report = classification_report(y_true, y_pred,
                                                             digits = self._digits)

    def __assert_equal_length(self, y_true, y_pred) -> None:
        assert (len(y_true) == len(y_pred)), "The length of the true labels is not the same as the pred labels."
    
    def __check_equal_number_of_classes(self, y_true, y_pred) -> None:
        if len(set(y_true)) != len(set(y_pred)):
            warnings.warn("WARNING: The number of classes in the true labels and the predicted labels is not the same.")
        
    def __generate_name_to_idx_dict(self):
        self.name_to_idx = dict(enumerate(self.y_true_names))
        self.name_to_idx = dict([(value, key) for key, value in self.name_to_idx.items()])

    def __prettify_confusion_matrix(self, confusion_matrix) -> str:
        """Prints out the confusion matrix with labels to console.
        Args:
            confusion_matrix: Uses the confusion matrix to format it in a pretty format. 
        Returns:
            str: Returns a pretty printed confusion matrix.
        """
        pandas_cf = pd.DataFrame(confusion_matrix,
                                 index = self.y_true_names,
                                 columns = self.y_true_names).round(self._digits)
        print(pandas_cf)

    def _setup_aucroc(self, y_softmax_probs: list):
        """A protected method to setup the auc-roc plot.

        Args:
            y_softmax_probs (list): This is a list of the predicted probabilities of the data. This expects a list that contains the softmax probs of all classes of all predictions.
        """
        # Much code for this function is adapted from
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_true_as_idx = np.array([self.name_to_idx.get(x) for x in self.y_true])
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_as_idx == i, y_softmax_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self.n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        self._fpr = fpr
        self._tpr = tpr
        self._roc_auc = roc_auc
        
    def get_metrics_dictionary(self) -> dict:
        """Returns a dictionary denoting the metrics of the model. 
        It is the accessible version of classification_report()
        
        Returns:
            dict: A dictionary containing the metrics of the model with respect to the classes that it predicts.
        """
        return classification_report(self.y_true, self.y_pred, 
                                     digits = self._digits, 
                                     output_dict = True)

    def classification_report(self) -> None:
        """Prints out the classification report to console.
        """
        print(self.__classification_report)
    
    def confusion_matrix(self) -> None:
        """Prints out the confusion matrix to console.
        """
        self.__prettify_confusion_matrix(self.__confusion_matrix)

    def norm_confusion_matrix(self, along = 'row') -> None:
        """Prints out a normalized confusion matrix to console, either normalized along the rows or columns.

        Args:
            along (str, optional): Takes 'row' or 'col' as arguments. Defaults to 'row'.
            The user specifies if the confusion matrix should be normalized along its rows or columns. 
        """
        AXIS_NUM = 1 if along == 'row' else 0
        DENOMINATOR = (self.__confusion_matrix.sum(axis = AXIS_NUM)[:, np.newaxis])
        if AXIS_NUM:
            NUMERATOR = self.__confusion_matrix.astype('float')
            norm_conf_mat = (NUMERATOR / DENOMINATOR)
        else:
            NUMERATOR = self.__confusion_matrix.T.astype('float')
            norm_conf_mat = (NUMERATOR / DENOMINATOR).T
        rounded_matrix = np.around(norm_conf_mat, decimals = self._digits)
        self.__prettify_confusion_matrix(rounded_matrix)

    def plot_roc_curve(self, y_softmax_probs: list, save = True,
                       image_filepath = os.getcwd(), 
                       image_filename = "roc_curve", 
                       class_line_width = 1,
                       macro_average_line_width = 3,
                       color_list = []) -> None:
        """This plots the roc curve in a new window and saves it to file if the 
        user so chooses. 

        Args:
            y_softmax_probs (list): This is a list of the predicted probabilities of the data. This expects a list that contains the softmax probs of all classes of all predictions.
            save (bool, optional): Save the generated image to file? Defaults to True.
            image_filepath (str, optional): The path to which the image is saved. Defaults to os.getcwd().
            image_filename (str, optional): The name of the image file. Defaults to "roc_curve".
            class_line_width (int, optional): Controls how wide the lines the multiple classes should be. Defaults to 1.
            macro_average_line_width (int, optional): Controls how wide the dashed line of the macro average of the ROC curves is. Defaults to 3.
            color_list (list, optional): A list of colors. Length should be the same as the number of classes. Defaults to []. If the user doesn't specify a color list, self._distinct_colors will be used, which is automatically generated.
        """
        
        self._setup_aucroc(y_softmax_probs)
        if len(color_list) == 0:
            color_list = self._distinct_colors
        # Much code for this function is adapted from
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        class_line_width = class_line_width
        macro_average_line_width = macro_average_line_width
        plt.figure()
        plt.plot(
            self._fpr["macro"],
            self._tpr["macro"],
            label = "Macro-Avg, (AUC = {0:0.3f})".format(self._roc_auc["macro"]),
            color = "navy",
            linestyle = ":",
            linewidth = macro_average_line_width)
        for i, color in zip(range(len(self.y_true)), color_list):
            plt.plot(
                self._fpr[i],
                self._tpr[i],
                color = color,
                lw = class_line_width,
                label = "{0}, (AUC = {1:0.3f})".format(self.y_true_names[i], self._roc_auc[i]))
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
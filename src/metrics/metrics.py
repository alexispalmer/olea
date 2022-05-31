import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
# precision_recall_fscore_support, auc
import numpy as np

class Metrics:
    def __init__(self, y_true: list, y_pred: list) -> None:
        """_summary_

        Args:
            y_true (list): A list containing the true labels of the data.
            y_pred (list): A list containing the predicted labels of the data.
        """
        self.__assert_equal_length(y_true, y_pred)
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_true_names = list(set(y_true))
        self.y_true_names.sort()
        self.y_pred_names = list(set(y_pred))
        self.y_pred_names.sort()
        self._digits = 4
        self.__confusion_matrix = confusion_matrix(y_true, y_pred)
        self.__classification_report = classification_report(y_true, y_pred,
                                                             digits = self._digits)

    def __assert_equal_length(self, y_true, y_pred) -> None:
        assert (len(y_true) == len(y_pred)), "The length of the true labels is not the same as the pred labels."

    def classification_report(self) -> str:
        """_summary_

        Returns:
            str: Returns a pretty printed classification report.
        """
        print(self.__classification_report)
    
    def confusion_matrix(self) -> None:
        """_summary_
        This prints the confusion matrix out to console.
        """
        self.__prettify_confusion_matrix(self.__confusion_matrix)

    def __prettify_confusion_matrix(self, confusion_matrix) -> None:
        """_summary_
        Args:
            confusion_matrix: Uses the confusion matrix to format it in a pretty format. 
        """
        pandas_cf = pd.DataFrame(confusion_matrix,
                                 index = self.y_true_names,
                                 columns = self.y_true_names).round(self._digits)
        print(pandas_cf)

    def norm_confusion_matrix(self, along = 'row') -> None:
        """_summary_

        Args:
            along (str, optional): Takes 'row' or 'col' as arguments. Defaults to 'row'.
            The user specifies if the confusion matrix should be normalized along its rows or columns. .
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

    def get_metrics_dictionary(self) -> dict:
        """_summary_
        
        Returns:
            dict: A dictionary containing the metrics of the model with respect to the classes that it predicts.
        """
        return classification_report(self.y_true, self.y_pred, 
                                     digits = self._digits, 
                                     output_dict = True)
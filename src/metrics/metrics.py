from sklearn.metrics import confusion_matrix, classification_report, auc
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
        
        self.digits = 4
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        self.__classification_report = classification_report(y_true, y_pred,
                                                             digits = self.digits)

    def __assert_equal_length(self, y_true, y_pred) -> None:
        assert (len(y_true) == len(y_pred)), "The length of the true labels is not the same as the pred labels."

    def classification_report(self) -> str:
        """_summary_

        Returns:
            str: Returns a pretty printed classification report.
        """
        print(self.__classification_report)

    def norm_confusion_matrix(self, along = 'row') -> np.ndarray:
        """_summary_

        Args:
            along (str, optional): Takes 'row' or 'col' as arguments. Defaults to 'row'.
            The user specifies if the confusion matrix should be normalized along its rows or columns. 

        Returns:
            np.ndarray: The normalized confusion matrix as a numpy array.
        """
        AXIS_NUM = 1 if along == 'row' else 0
        DENOMINATOR = (self.confusion_matrix.sum(axis = AXIS_NUM)[:, np.newaxis])
        if AXIS_NUM:
            NUMERATOR = self.confusion_matrix.astype('float')
            norm_conf_mat = (NUMERATOR / DENOMINATOR)
        else:
            NUMERATOR = self.confusion_matrix.T.astype('float')
            norm_conf_mat = (NUMERATOR / DENOMINATOR).T
        rounded_matrix = np.around(norm_conf_mat, decimals = self.digits)
        return rounded_matrix

    def get_metrics_dictionary(self) -> dict:
        """_summary_
        
        Returns:
            dict: A dictionary containing the metrics of the model with respect to the classes that it predicts.
        """
        return classification_report(self.y_true, self.y_pred, 
                                     digits = self.digits, 
                                     output_dict = True)
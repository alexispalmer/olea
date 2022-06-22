from src.data.dso import DatasetSubmissionObject
from src.viz.viz import plot_bar_graph


class Generic(object) : 

    @staticmethod
    def check_substring(substring:str, submission:DatasetSubmissionObject) : 
        data, groundtruth, prediction = submission.filter_submission('Text', lambda x : substring in x)
        return data, groundtruth, prediction

    @staticmethod
    def str_len_analysis(submission:DatasetSubmissionObject) : 
        pass

    @staticmethod
    def check_anno_agreement(submission: DatasetSubmissionObject, 
                             anno_columns: list, 
                             preds: list, 
                             plot = True) -> pd.DataFrame:
        """This calculates the annotator agreement of the dataset.

        Args:
            submission (DatasetSubmissionObject): This is a DatasetSubmissionObject (which ultimately is a pd.DataFrame)

            anno_columns (list): Give the column names that contain the
            annotator values in the dataframe as a list, 
            eg: ["Anno1", "Anno2", "Anno3"].
            preds (list): Give the column name that contains the predicted 
            values in the dataframe as a list, eg: ["preds"].
            plot (bool, optional): Plot the annotator agreement as a bar plot. 
            Defaults to True.

        Returns:
            pd.DataFrame: Returns a dataframe that calculates the full 
            agreemeent, total correct predictions, total, and accuracy values 
            for the annotator data.
        """
        predictions = submission[preds]
        annotator_columns = submission[anno_columns]

        #full agreement is considered the "easy case"
        full_agree = annotator_columns[annotator_columns.eq(annotator_columns.iloc[:, 0], axis=0).all(axis=1)]
        n_full_agree = full_agree.shape[0]

        #not full agreement is considered the more difficult case"
        partial_agree = annotator_columns[~annotator_columns.loc[:].isin(full_agree.loc[:])].dropna()
        n_partial_agree = partial_agree.shape[0]

        #find totals
        labels = ["Full", "Partial"]
        totals = [n_full_agree, n_partial_agree]

        # Assumes the predictions are in the same order as the dataframe
        # would be unusual if they weren't.
        full_preds = [predictions[i] for i in full_agree.index]
        partial_preds = [predictions[i] for i in partial_agree.index]
        
        correct_predictions = [full_preds, partial_preds]

        #include the rest of the cold data
        # full_agree_df = submission.loc[full_agree.index, :]
        # partial_agree_df = submission.loc[partial_agree.index,:]

        # plot the bar graph
        if plot:
            plot_bar_graph(labels, totals, correct_predictions, 
                            title = "Predictions on Text with Full vs Partial Annotator Agreement")
        
        #create df to store results
        percents = [a/b if b else 0 for a,b in zip(correct_predictions, totals)]
        results = pd.DataFrame({"Full Agreement" : ["Y","N"],"Total_Correct_Predictions": correct_predictions, "Total": totals,"Accuracy": percents})

        return results
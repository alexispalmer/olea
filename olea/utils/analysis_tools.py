from olea.metrics.metrics import Metrics
import numpy as np
import pandas as pd

def get_examples(submission,column,results, sort_list= False):
    """pull examples where model output label does not line up with true label to illustrate specific cases. Adds one text examples from each value present in the specified column to results data

    Args:
        df (df): data to pull from, containing textual data
        column (string): column name to evaluate on
        results (df): df containing information for each classification, (created by all analysis functions)
        sort_list (boolean): if the values need to be sorted for presentation (currently just used by text_length analysis)
    Returns:
        df: updated results 
    """
    df = submission.submission
    off_col = submission.gold_column
    preds = submission.prediction_column
    
    column_vals = results[column]
    if sort_list:
        column_vals = sorted(column_vals,key=lambda x: float(x.split('-')[0].replace(',','')))
    examples= ["" for x in range(results.shape[0])]
    examples_pred = ["" for x in range(results.shape[0])]
    examples_gold= ["" for x in range(results.shape[0])]
    
    incorrect_df = df[df[preds] != df[off_col]]
    
    for i in range(len(column_vals)):
        incorrect_values = incorrect_df[incorrect_df[column] == column_vals[i]]
        if incorrect_values.empty:
            examples[i] = ""
            examples_pred[i] = ""
            examples_gold[i] = ""
        else:
            example = incorrect_values.sample(1)
            examples[i] = example[submission.text_column].iloc[0]
            examples_pred[i] = example[submission.prediction_column].iloc[0]
            examples_gold[i] = example[submission.gold_column].iloc[0]
        
    results["Example with Incorrect Classification"] = examples
    results["Example Predicted Label"] = examples_pred
    results["Example Gold Label"] = examples_gold
    
    return results
    

def get_metrics(submission, column, cats_based_on_labels = False):

    """Returns metrics information for each categroy specified by column if a column is specified, otherwise it returns metrics over the whole dataset
    
    Args:
        df (df): data to pull from, containing textual data
        column (string): column name to evaluate on, defaults to None. If None, metrics are calculated over the whole dataset
        off_col (string): label of ground truth column
        remove_cat_labels (boolean): if nonsensical labels would be present in results (currently just used by fine-grained category analysis for COLD)
    Returns:
        df: metrics information
    
    """
    off_col = submission.gold_column
    preds = submission.prediction_column
    df = submission.submission
    
    if column == off_col:
        #coarse metrics over entire dataset
        # my_metric = Metrics(df[off_col],df["preds"])
        metrics_dict = Metrics.get_metrics_dictionary(y_true = df[off_col], 
        y_pred = df[preds])
        del metrics_dict["accuracy"]
        metrics_df = pd.DataFrame(metrics_dict)
    else:
        #metrics by category specified by column
        metrics_dict = {}
        column_vals = np.unique(df[column].astype(str))
        for value in column_vals:
            df_subset = df.loc[df[column] == value]
            # my_metric = Metrics(df_subset[off_col],df_subset["preds"])
            m_dict = Metrics.get_metrics_dictionary(y_true = df_subset[off_col], 
                                                    y_pred = df_subset[preds])
            del m_dict["accuracy"] #remove accuracy metric, can be viewed elsewhere
            
            if cats_based_on_labels:
                m_dict = __remove_nonsensical_labels(m_dict, df_subset,off_col,np.unique(df[off_col]))
            metrics_dict[value] = m_dict
    
        metrics_df = __generate_metrics_df(metrics_dict,column)
    #return df for dataframe version
    return metrics_df

def __generate_metrics_df(metrics_dict,column):
    """Converts dict of dicts to double-indexed df for better viewing of metrics

    Args:
        metrics_dict (dict): metrics info 
        column (str): Name of Category for Metrics -corresponds to a column name

    Returns:
        df: metrics info in dataframe form
    """
    #Double index version
    reformed_dict = {}
    for outerKey, innerDict in list(metrics_dict.items()):
        for innerKey, values in innerDict.items():
            reformed_dict[(outerKey, innerKey)] = values
    
    multi_indices = pd.MultiIndex.from_tuples(reformed_dict, names=[column, "Metrics"])
    metrics_df = pd.DataFrame.from_dict(reformed_dict.values())
    metrics_df = metrics_df.set_index(multi_indices)
    
    return metrics_df

def __remove_nonsensical_labels(m_dict, df_subset,off_col,off_labels):
    """changes nonsensical metrics label information to dashes. Nonsensical labels arrise when
       in metrics when the labels of the category that is being analyzed are derived partly from the Offensive 
       Label itself. For example in COLD, the fine-grained categorie that include "Reclaimed Slur" is in definition 
       Non-offensive, so this function removes metrics for the category "reclaimed slur" that are offensive. This helps for the
       clarity


    Args:
        m_dict (dict): the metrics dict to remove 
        df_subset (df): subset of df containing only instances of the specific category
        off_col (string): ground truth column name for offensiveness 

    Returns:
        m_dict: m_dict with dashes where nonsensical information would be
    """
    #remove unnecessary labels when getting metrics for Fine-grained categories
    label_to_keep = df_subset[off_col].iloc[0]
    if label_to_keep == off_labels[0]:
        label_to_drop = off_labels[1]
    else:
        label_to_drop = off_labels[0]
    if label_to_drop in m_dict:
        for key in m_dict[label_to_drop]:
            m_dict[label_to_drop][key] = '-'
        
    return m_dict

# def get_plotting_info_create_col(df1,df2, new_feature,off_col):
#     """prepare two dataframes representing (Y/N) binary distinction of a category for metrics and plotting calculation. 
#     Creates a new column of that category for metrics and plotting usage. Passes df with new column into get_plotting_info_from_col 
#     for calculation.

#     Args:
#         df1 (df): one dataframe containing "Y" instances of whatever category. (ie. Instances that contain a hashtag)
#         df2 (df): one dataframe containing "N" non-instances of whatever category.  (ie. Instances that do not contain a hashtag)
#         new_feature (string): name of the new feature (ie. Presence of hashtag)
#         off_col (string): offesnive column

#     Returns:
#        totals  (list): total instances corresponding to each category
#        correct_predictions_n  (list): total instances corresponding to each category that the model predicted correctly
#        results  (df): summarization of results, corresponds to plotted information
#        full_df  (df): df with the new column - necessary for plotting and retrieving examples
#     """
#     df1[new_feature] = "Y"
#     df2[new_feature] = "N"
#     full_df = df1.merge(df2,"outer")

#     totals, correct_predictions_n,results= get_plotting_info_from_col(full_df, new_feature, off_col)

    # totals = [df1.shape[0], df2.shape[0]]
    # #find correct predicitons
    # correct_preds_1 = df1[(df1['preds'] == df1[off_col])]
    # correct_preds_2 = df2[(df2['preds'] ==df2[off_col])]
    # correct_predictions_n = [correct_preds_1.shape[0], correct_preds_2.shape[0]]
    
    
    
    # percents = [a/b if b else 0 for a,b in zip(correct_predictions_n, totals)]
    # results = pd.DataFrame({"Full Agreement" : ["Y","N"],"Total_Correct_Predictions": correct_predictions_n, "Total": totals,"Accuracy": percents})
    
   # return totals, correct_predictions_n, results, full_df

def get_plotting_info_from_col(submission, feature):
    """calculates info from a dataframe for metrics and plotting usage using labels from the feature parameter.

    Args:
        df (df): full dataframe with text,off label, and contains a column with category labels for each isntance
        feature (string): name of the feature
        off_col (string): offesnive column

    Returns:
       totals  (list): total instances corresponding to each category
       correct_predictions_n  (list): total instances corresponding to each category that the model predicted correctly
       results  (df): summarization of results, corresponds to plotted information. Contains totals, correct_predictions, and accuracies
    """
    df = submission.submission
    off_col = submission.gold_column
    preds = submission.prediction_column
    
    correct_preds = df[(df[preds] == df[off_col])]
    #get totals
    totals = df[feature].value_counts()
    correct_predictions_n = correct_preds[feature].value_counts()
    
    total_df = pd.concat([totals,correct_predictions_n],axis=1)
    totals = total_df.iloc[:,0]
    correct_predictions_n = total_df.iloc[:,1]

    
    Accuracy = correct_preds[feature].value_counts()/df[feature].value_counts()
    results = pd.DataFrame({"Total":totals, "Total_Correct_Predictions": correct_predictions_n, "Accuracy": Accuracy})
    results = results.reset_index()
    results = results.rename(columns = {'index':feature})
    
    return results
    

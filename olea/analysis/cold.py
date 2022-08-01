from olea.data.dso import DatasetSubmissionObject
from olea.utils.analysis_tools import get_metrics, get_examples
from olea.utils.analysis_tools import get_plotting_info_from_col
from olea.viz.viz import plot_bar_graph
import numpy as np

class COLDAnalysis(object) : 
    
        
    @classmethod
    def _run_analysis_on_functionality(cls,submission:DatasetSubmissionObject, on:str,plot, show_examples,savePlotToFile) :
        """helper function for running analysis on a specific column. Returns two dataframes. plot_info corresponds to 
            information that is plotted, number of offensive/non offensive instances for each category in "on" as well as
            accuracy of model. Metrics returns the classification report for each category specified on "on"

        Args:
            submission (COLDSubmissionObject): submission object to run analysis on
            on (str): column name in submission.submission dataframe to run analysis on
            plot (boolean): to plot results or not
            show_examples (boolean): to return examples or not
            savePlotToFile (str): File name for saving plot, empty string will not save a plot

        Returns:
            plot_info (df) : results corresponding to plotted information:(total instances for each category,
                             total correctly predicted, and accuracy)
            metrics (df): classification report for each category
        """
        cats_based_on_labels =False
        rot = 0
        if on == "Cat":
            cats_based_on_labels =True
            rot = 45
            
        #labels = np.unique(submission.submission[on])
        plot_info = get_plotting_info_from_col(submission, feature = on)
         
         # plot the bar graph
        if plot:
            plot_bar_graph(plot_info, savePlotToFile, title = str("Predictions on " + on),rot = rot)
         #get examples
        if show_examples:
            plot_info = get_examples(submission, on, plot_info)
         
        metrics = get_metrics(submission, on,cats_based_on_labels)    
        
        return plot_info, metrics
            
    @classmethod
    def analyze_on(cls, submission:DatasetSubmissionObject, on:str,plot=True,show_examples = False,savePlotToFile = "") : 
        """function for running analysis on a specific column, and plots results if specified. Returns two dataframes. 
            plot_info corresponds to information that is plotted, number of offensive/non offensive instances for each category 
            in "on" as well as
            accuracy of model. Metrics returns the classification report for each category specified on "on"

        Args:
            submission (COLDSubmissionObject): submission object to run analysis on
            on (str): column name in submission.submission dataframe to run analysis on
            plot (boolean): to plot results or not
            show_examples (boolean): to return examples or not
            savePlotToFile (str): File name for saving plot, empty string will not save a plot

        Returns:
            plot_info (df) : results corresponding to plotted information:(total instances for each category,
                             total correctly predicted, and accuracy)
            metrics (df): classification report for each category
        """
        return cls._run_analysis_on_functionality(submission, on, plot,show_examples,savePlotToFile)

        

    # @staticmethod
    # def coarse_analysis(cold_submission:DatasetSubmissionObject, off_col = "Off",plot=True,show_examples=False): 
    #     """check how model predicts on instnaces labeled offensive vs non offensive

    #     Args:
    #         cold_submission (DatasetSubmissionObject): submission object to use, containing df
    #         plot (boolean): to plot or not to plot
    #         show_examples (boolean): to show examples of instances the model predicted incorrectly or not
    #                                 (WARNING: LIKELY CONTAINS OFFENSIVE LANGUAGE)
    #         off_col (str): ground truth column name 
    #     Returns:
    #         results (df) : results corresponding to plotted information:(total instances for each category,
    #                          total correctly predicted, and accuracy)
    #         metrics (df): metrics information for each category
    #     """ 
    #     # groundtruth, predictions = cold_submission.submission['Off'] , cold_submission.submission['preds']
    #     labels = np.unique(cold_submission.submission[off_col])
    #     totals, correct_predictions_n, results = get_plotting_info_from_col(cold_submission.submission, off_col, off_col)
        
    #     # plot the bar graph
    #     if plot:
    #         plot_bar_graph(labels, totals, correct_predictions_n, 
    #                         title = "Coarse Predictions of Offensiveness")
    #     #get examples
    #     if show_examples:
    #         results = get_examples(cold_submission.submission,off_col,results,off_col)
        
    #     metrics = get_metrics(cold_submission.submission,column= None, off_col=off_col)    
    #     return results, metrics
    #     # m = Metrics(groundtruth, predictions)
    #     # return m.get_metrics_dictionary()


    # @staticmethod
    # def categorical_analysis(cold_submission:DatasetSubmissionObject, category:str,plot=True,show_examples = False,off_col = "Off",cats_based_on_labels =True) : 
    #     """check how model predicts on different labels of specific category already in the df

    #     Args:
    #         cold_submission (DatasetSubmissionObject): submission object to use, containing df
    #         category (str): column name
    #         plot (boolean): to plot or not to plot
    #         show_examples (boolean): to show examples of instances the model predicted incorrectly or not
    #                                 (WARNING: LIKELY CONTAINS OFFENSIVE LANGUAGE)
    #         off_col (str): ground truth column name 
    #     Returns:
    #         results (df) : results corresponding to plotted information:(total instances for each category,
    #                          total correctly predicted, and accuracy)
    #         metrics (df): metrics information for each category
    #     """ 
    #     totals, correct_predictions_n,results = get_plotting_info_from_col(cold_submission.submission, category, off_col)
    #     labels = np.unique(cold_submission.submission[category])
        
    #     # plot the bar graph
    #     if plot:
    #         plot_bar_graph(labels, totals, correct_predictions_n, 
    #                         title = "Predictions on Text of " + category)
    #     #get examples
    #     if show_examples == True:
    #         results = get_examples(cold_submission.submission,category,results,off_col=off_col)
        
    #     #get_metrics
    #     metrics = get_metrics(cold_submission.submission, category ,off_col,cats_based_on_labels)
        
    #     return results, metrics   
    
    
    
    

        
        # groundtruth, predictions, category_labels = cold_submission.submission['Off'], cold_submission.submission['preds'], cold_submission.submission[category]
    
        # gt_dict = {}
        # pt_dict = {}
        
        # for gt, pt, ct in zip(groundtruth, predictions, category_labels) : 
    
        #     if ct in gt_dict : 
        #         gt_dict[ct].append(gt)
        #         pt_dict[ct].append(pt)
        #     else : 
        #         gt_dict[ct] = [gt]
        #         pt_dict[ct] = [pt]
    
        # result_dict = {}
            
        # for ct in gt_dict.keys() :
        #     m = Metrics(gt_dict[ct] , pt_dict[ct])  
        #     result_dict[ct] = m.get_metrics_dictionary()
                
            
        # return result_dict

if __name__ == '__main__' : 

    from olea.data.cold import COLD, COLDSubmissionObject
    from olea.analysis.cold import COLDAnalysis
    import numpy as np
    from olea.utils.preprocess_text import PreprocessText as pt

    cold = COLD()

    dataset = cold.data()
    messages = list(dataset['Text'])
    pre = pt.execute(messages)
    dataset['pre'] = pre

    num_preds = dataset.shape[0]
    yn_preds = np.random.choice(['Y' , 'N'], size=num_preds)
    bool_preds = np.random.choice([True, False], size=num_preds)

    map = {True : 'Y' , False:'N'}

    print('Yes-No Preds')

    submission = cold.submit(dataset, bool_preds, map=map)

    coarse_results = COLDAnalysis.analyze_on(submission,"Off",plot=True,show_examples = True)
    nom_results = COLDAnalysis.analyze_on(submission,"Nom",plot=True,show_examples = True)
    cat_results = COLDAnalysis.analyze_on(submission,"Cat", plot= True, show_examples = False)

    #print(COLDAnalysis.categorical_analysis(submission, category='Nom'))














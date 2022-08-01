from olea.data.dso import DatasetSubmissionObject
from olea.utils.twitteraae.code import detection 
from typing import Union, List
from olea.viz.viz import plot_bar_graph
from olea.viz.viz import plot_histogram, histogram_values
from olea.utils.analysis_tools import get_metrics, get_examples
from olea.utils.analysis_tools import get_plotting_info_from_col
import pandas as pd
import numpy as np

class Generic(object) : 
    
    
    '''TO DO:
        Figure out how to incorporate a list of columns
    '''
    @classmethod
    def _run_analysis_on(cls, submission:DatasetSubmissionObject, 
                        on:str, 
                        target_column:str,plot, show_examples,savePlotToFile) : 
        
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

         #labels = np.unique(submission.submission[on])
        plot_info = get_plotting_info_from_col(submission, feature = on)
          
          # plot the bar graph
        if plot:
             plot_bar_graph(plot_info, savePlotToFile, title = str("Predictions on " + on))
          #get examples
        if show_examples:
             plot_info = get_examples(submission, on, plot_info)
          
        metrics = get_metrics(submission, on)    
         
        return plot_info, metrics

    @classmethod
    def analyze_on(cls, submission:DatasetSubmissionObject, 
                    features:Union[str, List[str]],plot=True,show_examples=False, savePlotToFile= ""): 
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
        
        
        return cls._run_analysis_on(submission, 
                                    features, 
                                    submission.gold_column, 
                                    plot,
                                    show_examples,
                                    savePlotToFile)

    @classmethod
    def check_substring(cls, submission:DatasetSubmissionObject,
                        substring:str,
                        plot=True,
                        show_examples=False,
                        savePlotToFile= ""):
        """check how model predicts on instances containing a specific substring vs without that substring.

        Args:
            submission (DatasetSUbmissionObject): submission object to use, containing df
            substring (str): stubstring to search for 
            plot (boolean): to plot or not to plot
            show_examples (boolean): to show examples of instances the model predicted incorrectly or not
                                    (WARNING: LIKELY CONTAINS OFFENSIVE LANGUAGE)
            savePlotToFile (str): File name for saving plot, empty string will not save a plot
        Returns:
            plot_info (df) : results corresponding to plotted information:(total instances for each category,
                             total correctly predicted, and accuracy)
            metrics (df): classification report for each category
        """ 
        #find instances of substring/ vs no substring
        df1 = submission.submission.loc[submission.submission[submission.text_column].str.contains(substring)].copy()
        df2 = submission.submission.loc[~submission.submission[submission.text_column].str.contains(substring)].copy()
        
        #create new column
        new_feature = ''.join(["Containing ", '\'',substring, '\''])
        labels = [new_feature, str("Not " + new_feature)]
        df1[new_feature] = labels[0]
        df2[new_feature] = labels[1]
        submission.submission = df1.merge(df2,"outer")

        #run analysis
        return cls._run_analysis_on(submission, 
                                    new_feature, 
                                    submission.gold_column, 
                                    plot,
                                    show_examples,
                                    savePlotToFile)
     


    @classmethod
    def aave(cls,submission:DatasetSubmissionObject,
             threshold:float = 0.5,
             plot=True,
             show_examples=False,
             savePlotToFile= ""
             ):
        """Check how model predicts on instances that are written using African American Vernacular English. The scores 
            are calculated using the TwitterAAE model created by (Blodgett et. al 2016). Further information can be found at
            http://slanglab.cs.umass.edu/TwitterAAE/. These scores represent an inference of the *proportion* of words in the 
            text that come from a demographically-associated language/dialect.
        
        Args:
            submission (DatasetSUbmissionObject): submission object to use, containing df
            threshold (float): threshold from (0-1) as cutoff for considering a text to be AAVE
            plot (boolean): to plot or not to plot
            show_examples (boolean): to show examples of instances the model predicted incorrectly or not
                                    (WARNING: LIKELY CONTAINS OFFENSIVE LANGUAGE)
            savePlotToFile (str): File name for saving plot, empty string will not save a plot
        Returns:
            plot_info (df) : results corresponding to plotted information:(total instances for each category,
                             total correctly predicted, and accuracy)
            metrics (df): classification report for each category
        """ 
        #get aave values
        aave_values = detection.get_aave_values(submission)
        submission.submission["AAVE"] = aave_values
        df1 = submission.submission.loc[submission.submission['AAVE'] >= threshold].copy()
        df2 = submission.submission.loc[submission.submission['AAVE'] < threshold].copy()
        
        #create new column
        new_feature = "AAVE-thresh >= " + str(threshold) #Add in new feature with labels for metrics
        labels = ["Y", "N"]
        df1[new_feature] = labels[0]
        df2[new_feature] = labels[1]
        submission.submission = df1.merge(df2,"outer")
        
       
        return cls._run_analysis_on(submission, 
                                    new_feature, 
                                    submission.gold_column, 
                                    plot,
                                    show_examples,
                                    savePlotToFile)
        
        
                

    @staticmethod
    def str_len_analysis(submission:DatasetSubmissionObject,
                         analysis_type = "character",
                         hist_bins = 10,
                         plot=True,
                         show_examples=False,
                         savePlotToFile= ""
                         ):
        """Analyze and plot how the model performs on instances of different lengths using a histogram.

        Args:
            submission (DatasetSUbmissionObject): submission object to use, containing df
            hist_bins (int): number of bins to use for the histogram
            analysis_type (str) : options are ["charcter", "word", "word_len"] for character count, word count, or avg characters per word
            plot (boolean): to plot or not to plot
            show_examples (boolean): to show examples of instances the model predicted incorrectly or not
                                    (WARNING: LIKELY CONTAINS OFFENSIVE LANGUAGE)
            savePlotToFile (str): File name for saving plot, empty string will not save a plot
        Returns:
            plot_info (df) : results corresponding to plotted information:(total instances for each category,
                             total correctly predicted, and accuracy)
            metrics (df): classification report for each category
        """  
        #add new feature
        if analysis_type == "character":
            new_feature = "Character Count"
            submission.submission[new_feature] = submission.submission[submission.text_column].apply(len)
        elif analysis_type == "word":
            new_feature = 'Word Count'
            submission.submission[new_feature] = submission.submission[submission.text_column].str.split().apply(len)
        else:
            new_feature = "Average Characters per Word"
            submission.submission[new_feature] = [np.average([len(i) for i in row])for row in submission.submission[submission.text_column].str.split()]
       
        correct_preds = submission.submission[(submission.submission[submission.prediction_column] ==submission.submission[submission.gold_column])]
            
        bins, bin_vals, bin_vals_correct = histogram_values(submission.submission[new_feature],
                                                correct_preds[new_feature])
        #combine histogram info for printing
        ranges = []
        percents = []
        i = 0
        while i <len(bins)-1:
            ranges.append(str(np.fix(bins[i])) + " - " +str(np.fix(bins[i+1])))
            if bin_vals[i]:
                percents.append(bin_vals_correct[i]/bin_vals[i])
            else:
                percents.append(0)
            i+=1
            
        if plot:
            plot_histogram(title = str("Predictions on " + new_feature), 
                           hist_bins = hist_bins,
                           xlabel = "Text Length",
                           list_of_values =  submission.submission[new_feature],
                           correct_preds = correct_preds[new_feature],
                           accuracy = percents, savePlotToFile="")
        #reformat bins for clarity
        new_feature_list = []
        for tl in submission.submission[new_feature]:
            for i in range(len(bins)-1):
                #this line looks complicated, but just accounts for the last bin of histogram to be inclusive 
                if (tl>= bins[i] and tl<bins[i+1] or (tl>= bins[i] and tl==(bins[i+ 1]) and i==len(bins)-2)):
                    new_feature_list.append(ranges[i])
        
        submission.submission[new_feature] = new_feature_list
        
        results = pd.DataFrame({new_feature : ranges, "Total": bin_vals, "Total_Correct_Predictions": bin_vals_correct, "Accuracy": percents})
        
        #find examples
        if show_examples:
            results = get_examples(submission, new_feature, results,sort_list = True)

        metrics = get_metrics(submission,new_feature)
        
        return results,metrics
    
    @classmethod
    def check_anno_agreement(cls,submission: DatasetSubmissionObject, 
                             anno_columns: list,
                             plot = True, show_examples=False,
                             savePlotToFile= "") -> pd.DataFrame:
        """how model predicts on instances where annotators fully agree in whether text is offensive ("Y","Y","Y") or ("N","N","N).
#             vs when there is partial agreement. This should indicate performance on "easy" (full) vs "difficult" (partial) cases
#             Updates cold dataset with new feature column of full vs partial agreement

        Args:
            submission (DatasetSubmissionObject): This is a DatasetSubmissionObject (which ultimately is a pd.DataFrame)
            anno_columns (list): Give the column names that contain the annotator values in the dataframe as a list, 
                eg: ["Anno1", "Anno2", "Anno3"].
            plot (bool, optional): Plot the annotator agreement as a bar plot. Defaults to True.
            show_examples(bool,optional) : to show examples of instances the model predicted incorrectly or not
                                         (WARNING: LIKELY CONTAINS OFFENSIVE LANGUAGE)
            savePlotToFile (str): File name for saving plot, empty string will not save a plot

        Returns:
            plot_info (pd.DataFrame): Returns a dataframe that calculates the full 
            agreemeent, total correct predictions, total, and accuracy values 
            for the annotator data.
            metrics (df): classification report for each category


        """
        #full agreement is considered the "easy case"
        df1 = submission.submission.loc[submission.submission[anno_columns].eq(submission.submission[anno_columns].iloc[:, 0], axis=0).all(axis=1)].copy()
        #not full agreement is considered the more difficult case"
        df2 = submission.submission[~submission.submission.loc[:].isin(df1.loc[:])].copy().dropna()
        
        
        #create new column
        new_feature = "Full Agreement vs Partial Agreement"
        labels = ["Full", "Partial"]
        df1[new_feature] = labels[0]
        df2[new_feature] = labels[1]
        submission.submission = df1.merge(df2,"outer")
        
       
        return cls._run_analysis_on(submission, 
                                    new_feature, 
                                    submission.gold_column, 
                                    plot,
                                    show_examples,
                                    savePlotToFile)
        
        # totals,correct_predictions_n, results, full_df= get_plotting_info_create_col(
        #     full_agree,partial_agree, new_feature,off_col)
        
        # # plot the bar graph
        # if plot:
        #     plot_bar_graph(labels, totals, correct_predictions_n, 
        #                     title = "Predictions on Text with" + new_feature) 
        # #get examples
        # if show_examples == True:
        #     results = get_examples(full_df,new_feature,results,off_col=off_col,sort_list = False)
        
        #get_metrics
        # metrics = get_metrics(full_df, off_col,new_feature)
        # return results, metrics
    
    
    
    
if __name__ == '__main__' : 

    from olea.data.cold import COLD, COLDSubmissionObject
    from olea.analysis.generic import Generic
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

    # coarse_results = Generic.analyze_on(submission,"Off",plot=True,show_examples = True)
    # substr_results = Generic.check_substring(submission,"female",plot=True,show_examples= True)
    # aave_results = Generic.aave(submission,plot = True, show_examples = True)
    # anno_results = Generic.check_anno_agreement(submission, ["Off1","Off2","Off3"],plot = True, show_examples = True)
    str_len_results = Generic.str_len_analysis(submission, plot= True, show_examples = True )
    
   #nom_results = COLDAnalysis.analyze_on(submission,"Nom",plot=True,show_examples = True)
    #cat_results = COLDAnalysis.analyze_on(submission,"Cat", plot= True, show_examples = False)

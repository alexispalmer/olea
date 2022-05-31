import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Analysis:
    def __init__(self,cold):
        """Initialize analysis class

        Args:
            cold (dataframe): cold data including comun "pred" (predicted labels for each entry in form of 0/1)
        """
        self.cold = cold

    def check_string_len(self):
        """Analyze and plot how the model performs on instances of different lengths using a histogram
    
        Args:
           self(Analysis): instance of analysis class
        
        Returns:
            df: percent correct of model predictions on different text length ranges
        """
        
        cold = self.cold
        
        #update dataframe for this task
        cold['text_len'] = cold['Text'].apply(len)
        correct_preds = cold[(cold['pred'] ==cold['Q1'])]
        
        #plot histogram
        n,bins,_ = plt.hist(cold["text_len"],color="red",label = "Total",edgecolor='black')
        n_correct,_,_ = plt.hist(correct_preds["text_len"],bins =bins,color="blue", label ="Correct Prediciton",edgecolor='black')
        plt.legend(loc="upper right")
        plt.xlabel("Text Length")
        plt.title("Predictions on Different Text Lengths")
        plt.xticks(bins)
        plt.show()
        
        #combine histogram info for printing
        ranges = []
        percents = []
        i = 0
        while i <len(bins)-1:
            ranges.append(str(bins[i]) + " - " +str(bins[i+1]))
            if n[i]:
                percents.append(n_correct[i]/n[i])
            else:
                percents.append(0)
            i+=1
        
        str_len_results = pd.DataFrame({"Text_Length" : ranges, "Total": n, "Total_Correct_Predictions": n_correct, "Percent_Correct": percents})
        
        #print histogram information
        print(str_len_results.to_string())
        
        return str_len_results


    def check_substring(self,substring):
        """check how model predicts on instances containing a specific substring vs without that substrig

        Args:
            substring (string): substring to find in instances

        Returns:
            df: percent correct of model predictions on instances containing substring vs not
        """
        cold =self.cold      
        #find instances of hashtags/ vs no hashtags
        ss = cold[cold['Text'].str.contains(substring)]
        no_ss = cold[~cold['Text'].str.contains(substring)]
        
        #find correct predictions on hashtags vs no hashtags
        correct_preds_ss = ss[(ss['pred'] ==ss['Q1'])]
        correct_preds_nss = no_ss[(no_ss['pred'] ==no_ss['Q1'])]
        
        #find totals
        labels = [substring,str(str("No ") + substring)]
        totals = [ss.shape[0],no_ss.shape[0]]
        correct_predictions =[correct_preds_ss.shape[0], correct_preds_nss.shape[0]]
        
        #plot bar graph
        plt.bar(labels,totals, color="red",label = "Total",edgecolor='black')
        plt.bar(labels,correct_predictions,color="blue", label ="Correct Prediciton",edgecolor='black')
        plt.legend(loc="upper right")
        plt.title(str("Predictions on Text with " + "'"+ substring+ "'"))
        plt.show()
        
        #create df to store results
        percents = [a/b for a,b in zip(correct_predictions,totals)]
        ss_results = pd.DataFrame({str(substring + " Presence") : labels,"Total_Correct_Predictions": correct_predictions, "Total": totals,"Percent_Correct": percents})
        print(ss_results.to_string())
        return ss_results
    
    def check_confidence(self):
        pass

    def check_anno_agreement(self):
        pass

    
    


    

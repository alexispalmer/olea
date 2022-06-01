import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Analysis:
    def __init__(self,cold,show_examples):
        """Initialize analysis class

        Args:
            cold (dataframe): cold data including comun "pred" (predicted labels for each entry in form of 0/1)
            show_examples (boolean): if textual examples (potentially containing offensive language) should be displayed
        """
        
        self.cold = cold
        self.show_examples = show_examples

    def check_string_len(self):
        """Analyze and plot how the model performs on instances of different lengths using a histogram
    
        Args:
           self(Analysis): instance of analysis class
        
        Returns:
            df: percent correct of model predictions on different text length ranges 
            df: cold dataset with new feature of text length range for metrics usage
        """
        
        cold = self.cold
        
        #update dataframe for this task
        cold['text_len'] = cold['Text'].apply(len)
        correct_preds = cold[(cold['pred'] ==cold['OffMaj'])]
        
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
            ranges.append(str(np.fix(bins[i])) + " - " +str(np.fix(bins[i+1])))
            if n[i]:
                percents.append(n_correct[i]/n[i])
            else:
                percents.append(0)
            i+=1
        
        str_len_results = pd.DataFrame({"Text_Length" : ranges, "Total": n, "Total_Correct_Predictions": n_correct, "Percent_Correct": percents})
        
        #print histogram information
        print(str_len_results.to_string())
        
        #add new feature
        new_feature = "Text Length Bin"
        new_feature_list = []
        for tl in cold["text_len"]:
            for i in range(len(bins)-1):
                #this line looks complicated, but just accounts for the last bin of histogram to be inclusive 
                if (tl>= bins[i] and tl<bins[i+1] or (tl>= bins[i] and tl==(bins[i+ 1]) and i==len(bins)-2)):
                    new_feature_list.append(ranges[i])
        cold_text_len = cold
        cold_text_len[new_feature] = new_feature_list
                
        #print examples
        if self.show_examples:
             self.get_examples(cold_text_len,new_feature,sort_list = True)
        
        return str_len_results, cold_text_len


    def check_substring(self,substring):
        """check how model predicts on instances containing a specific substring vs without that substring

        Args:
            substring (string): substring to find in instances

        Returns:
            df: percent correct of model predictions on instances containing substring vs not
            df: cold dataset with new feature of 'contains substring' for metrics usage
        """
        cold =self.cold      
        #find instances of substring/ vs no substring
        ss = cold[cold['Text'].str.contains(substring)]
        no_ss = cold[~cold['Text'].str.contains(substring)]
        
        #Add in new feature
        new_feature = ''.join(["Contains ", '\'',substring, '\''])
        ss[new_feature] = "Y"
        no_ss[new_feature] = "N"
        
        #find correct predictions on hashtags vs no hashtags
        correct_preds_ss = ss[(ss['pred'] ==ss['OffMaj'])]
        correct_preds_nss = no_ss[(no_ss['pred'] ==no_ss['OffMaj'])]
        
        #find totals
        labels = [substring,str(str("No ") + substring)]
        totals = [ss.shape[0],no_ss.shape[0]]
        correct_predictions =[correct_preds_ss.shape[0], correct_preds_nss.shape[0]]
        
        #plot bar graph        
        title = str("Predictions on Text with " + "'"+ substring+ "'")
        self.plot_bar_graph(labels,totals,correct_predictions,title)
        
        #create df to store results
        percents = [a/b if b else 0 for a,b in zip(correct_predictions,totals)]
        ss_results = pd.DataFrame({str(substring + " Presence") : labels,"Total_Correct_Predictions": correct_predictions, "Total": totals,"Percent_Correct": percents})
        print(ss_results.to_string())
        
         #merge all back together
        cold_ss = ss.merge(no_ss,"outer")
        
        if self.show_examples:
            self.get_examples(cold_ss,new_feature,sort_list = False)
        
        return ss_results, cold_ss
    
    def check_confidence(self):
        pass

    def fleiss_agreement(self):
        pass
    
    def check_anno_agreement(self,num_annotators):
        """check how model predicts on instances where annotators fully agree in whether text is offensive ("Y","Y","Y") or ("N","N","N).
            vs when there is partial agreement. This should indicate performance on "easy" (full) vs "difficult" (partial) cases

        Args:
            num_annotators (int): number of annotators used to determine full vs partial agreement

        Returns:
            df: percent correct of model predictions on full vs partial annotator agreement
            df: cold dataset with new feature of 'Full Agreement' for metrics usage
        """
        cold = self.cold
         
        #find indices of full/ partial agreement
        off_agreements = cold[["Off1", "Off2","Off3"]]
        #full agreement is considered the "easy case"
        full_agree = off_agreements[off_agreements.eq(off_agreements.iloc[:, 0], axis=0).all(axis=1)]
        #not full agreement is considered the more difficult case"
        partial_agree = off_agreements[~off_agreements.loc[:].isin(full_agree.loc[:])].dropna()
        
        #include the rest of the cold data
        full_agree_cold = cold.loc[full_agree.index, :]
        partial_agree_cold = cold.loc[partial_agree.index,:]
        
        #Add in new feature 
        new_feature = "Full Agreement"
        full_agree_cold[new_feature] = "Y"
        partial_agree_cold[new_feature] = "N"
        
        #find correct predicitons
        correct_preds_full = full_agree_cold[(full_agree_cold['pred'] ==full_agree_cold['OffMaj'])]
        correct_preds_partial = partial_agree_cold[(partial_agree_cold['pred'] ==partial_agree_cold['OffMaj'])]
        
        #find totals
        labels = ["Full","Partial"]
        totals = [full_agree_cold.shape[0],partial_agree_cold.shape[0]]
        correct_predictions =[correct_preds_full.shape[0], correct_preds_partial.shape[0]]
        
        #plot bar graph
        title = "Predictions on Text with Full Annotator Agreement vs Partial Annotator Agreement"
        self.plot_bar_graph(labels,totals,correct_predictions,title)
        
        #create df to store results
        percents = [a/b if b else 0 for a,b in zip(correct_predictions,totals)]
        agree_results = pd.DataFrame({"Agreement Level" : labels,"Total_Correct_Predictions": correct_predictions, "Total": totals,"Percent_Correct": percents})
        print(agree_results.to_string())
        
        #merge all back together
        cold_agree = full_agree_cold.merge(partial_agree_cold,"outer")
        
        if self.show_examples:
             self.get_examples(cold_agree,new_feature,sort_list = False)
            
        return agree_results, cold_agree
    
    
    def category_performance():
        pass
        
    
    def plot_bar_graph(self,labels, totals, correct_predictions,title):
        """plots bar graph for different analyses

        Args:
            labels (_type_): labels to be used for x axis
            totals (_type_): number of total instances for each class
            correct_predictions (_type_): number of correctly predicted instances for each class
            title (_type_): title of graph
        """
        plt.bar(labels,totals, color="red",label = "Total",edgecolor='black')
        plt.bar(labels,correct_predictions,color="blue", label ="Correct Prediciton",edgecolor='black')
        plt.legend(loc="upper right")
        plt.title(title)
        plt.show()
        
    def get_examples(self,df,column,sort_list):
        """pull examples to illustrate specific cases. Prints out one text examples from each value present in the specified column.

        Args:
            df (df): data to pull from, containing text data
            column (string): column name to evaluate on
            sort_list (boolean): if the values need to be sorted for presentation (currently just used by text_length analysis)
        """
        column_vals = np.unique(df[column])
        if sort_list:
            column_vals = sorted(column_vals,key=lambda x: float(x.split('-')[0].replace(',','')))
        prints = []
        for i in column_vals:
            examples = df[df[column] == i]["Text"]
            example = np.random.choice(df[df[column] == i]["Text"], 1)
            print("Example of ",'"',column,'"',": ", i,"; ",example)
        
        
        
        

        
    
    


    

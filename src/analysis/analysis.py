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
        """Analyze and plot how the model performs on instances of different lengths using a histogram. Updates cold dataset with new feature columns for text length and text length bin
    
        Args:
           self(Analysis): instance of analysis class
        
        Returns:
            df: percent correct of model predictions on different text length ranges 
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
            
        #add new feature
        new_feature = "Text Length Bin"
        new_feature_list = []
        for tl in cold["text_len"]:
            for i in range(len(bins)-1):
                #this line looks complicated, but just accounts for the last bin of histogram to be inclusive 
                if (tl>= bins[i] and tl<bins[i+1] or (tl>= bins[i] and tl==(bins[i+ 1]) and i==len(bins)-2)):
                    new_feature_list.append(ranges[i])
        
        cold[new_feature] = new_feature_list
        
        #print histogram information
        results = pd.DataFrame({new_feature : ranges, "Total": n, "Total_Correct_Predictions": n_correct, "Percent_Correct": percents})
        #print examples
        if self.show_examples:
             results = self.get_examples(cold,new_feature,results, sort_list = True)
        
        print(results)
        return results


    def check_substring(self,substring):
        """check how model predicts on instances containing a specific substring vs without that substring. Updates cold dataset with new feature column of contains substring 

        Args:
            substring (string): substring to find in instances

        Returns:
            df: percent correct of model predictions on instances containing substring vs not
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
        labels = ["Y","N"]
        totals = [ss.shape[0],no_ss.shape[0]]
        correct_predictions =[correct_preds_ss.shape[0], correct_preds_nss.shape[0]]
        
        #plot bar graph        
        title = str("Predictions on Text with " + "'"+ substring+ "'")
        self.plot_bar_graph(labels,totals,correct_predictions,title,rot = 0, xlabel= new_feature)
        
        #create df to store results
        percents = [a/b if b else 0 for a,b in zip(correct_predictions,totals)]
        results = pd.DataFrame({new_feature : labels,"Total_Correct_Predictions": correct_predictions, "Total": totals,"Percent_Correct": percents})
        
        #merge all back together and update dataset
        cold_ss = ss.merge(no_ss,"outer")
        cold[new_feature] = cold_ss[new_feature]
        
        if self.show_examples:
            results = self.get_examples(cold,new_feature,results)
        
        print(results)
        return results
    
    def check_confidence(self):
        pass

    def fleiss_agreement(self):
        pass
    
    def anno_fine_grained(self):
        """Analyze and plot how the model performs on fine-grained annotator agreement based on how many annotators said the instance was offensive
        [Y,Y,Y] = 3 , [Y,N,N] = 0 
        Updates cold dataset with new feature columns for text length and text length bin
    
        Args:
           self(Analysis): instance of analysis class
        
        Returns:
            df: percent correct of model predictions on different offesniveness rankings
        """
        cold = self.cold
        #create annotation rankings
        rank = []
        for i in range(len(cold)):
            annos = [cold.Off1[i],cold.Off2[i],cold.Off3[i]]
            if all(i == "Y" for i in annos):
                rank.append(3)
            elif all(i == "N" for i in annos):
                rank.append(0)
            elif max(set(annos), key=annos.count) == "Y":
                rank.append(2)
            else:
                rank.append(1)
            
                
        #Add in new feature
        new_feature = 'Fine-Grained Annotations'
        cold[new_feature] = rank
        
        #find correct predictions
        correct_cold =  cold[(cold['pred'] == cold['OffMaj'])]
        
        #combine bar chat info into dataframe
        totals = cold[new_feature].value_counts().sort_index()
        correct = correct_cold[new_feature].value_counts()
        percent_correct = correct_cold[new_feature].value_counts()/cold[new_feature].value_counts()
        results = pd.DataFrame({"Total":totals, "Total_Correct_Predictions": correct, "Percent_Correct": percent_correct})
        results = results.reset_index()
        results = results.rename(columns = {'index':new_feature})
        
        
        #print examples
        if self.show_examples:
            results = self.get_examples(cold,new_feature,results)
        
        #plot bar graph
        self.plot_bar_graph(results[new_feature],results.Total, results["Total_Correct_Predictions"], "Predictions on Fine-Grained Annotations", rot = 0, xlabel = "Number of Annotators that Marked Instance Offesive")
        print(results)
        
        return results
        
        
    
    def check_anno_agreement(self,num_annotators):
        """check how model predicts on instances where annotators fully agree in whether text is offensive ("Y","Y","Y") or ("N","N","N).
            vs when there is partial agreement. This should indicate performance on "easy" (full) vs "difficult" (partial) cases
            Updates cold dataset with new feature column of full vs partial agreement

        Args:
            num_annotators (int): number of annotators used to determine full vs partial agreement

        Returns:
            df: percent correct of model predictions on full vs partial annotator agreement
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
        title = "Predictions on Text with Full vs Partial Annotator Agreement"
        self.plot_bar_graph(labels,totals,correct_predictions,title)
        
        #create df to store results
        percents = [a/b if b else 0 for a,b in zip(correct_predictions,totals)]
        results = pd.DataFrame({new_feature : ["Y","N"],"Total_Correct_Predictions": correct_predictions, "Total": totals,"Percent_Correct": percents})
        
        #merge all back together
        cold_agree = full_agree_cold.merge(partial_agree_cold,"outer")
        cold[new_feature] = cold_agree[new_feature]
        
        if self.show_examples:
             results = self.get_examples(cold_agree,new_feature,results)
        
        print(results)   
        return results
    
    
    def category_performance(self):
        """check how model predicts on the 10 different fine-gained subcategories described in COLD (Palmer et al, 2020)
            Updates cold dataset with new feature column of category for each instance

        Args:
            num_annotators (int): number of annotators used to determine full vs partial agreement

        Returns:
            df: percent correct of model predictions on full vs partial annotator agreement
        """
        cold =self.cold 
        
        #create list of categories
        cats = []
        for i in range(len(cold)):
            if cold.OffMaj[i] == "Y":
                if cold.SlurMaj[i] == "Y":
                    cats.append("offSlur")
                elif cold.NomMaj[i] == "Y" and cold.DistMaj[i] == "Y":
                    cats.append("offBoth")
                elif cold.DistMaj[i] == "Y":
                    cats.append("offDist")
                elif cold.NomMaj[i] == "Y":
                    cats.append("offNom")
                else:
                    cats.append("offOther")
            else:
                if cold.SlurMaj[i] == "Y":
                    cats.append("reclaimed")
                elif cold.NomMaj[i] == "Y" and cold.DistMaj[i] == "Y":
                    cats.append("nonBoth")
                elif cold.DistMaj[i] == "Y":
                    cats.append("nonDist")
                elif cold.NomMaj[i] == "Y":
                    cats.append("nonNom")
                else:
                    cats.append("nonNone")
        
        #Add in new feature
        new_feature = 'Category'
        cold['Category'] = cats
        
        # #find correct predictions on hashtags vs no hashtags
        correct_cold =  cold[(cold['pred'] == cold['OffMaj'])]
        
        #combine bar chat info into dataframe
        totals = cold[new_feature].value_counts()
        correct = correct_cold[new_feature].value_counts()
        percent_correct = correct_cold[new_feature].value_counts()/cold[new_feature].value_counts()
        results = pd.DataFrame({"Total":totals, "Total_Correct_Predictions": correct, "Percent_Correct": percent_correct})
        results = results.reset_index()
        results = results.rename(columns = {'index':new_feature})
        
        if self.show_examples:
            results = self.get_examples(cold,new_feature,results)
        
        print(results)
        #plot bar chart
        self.plot_bar_graph(results[new_feature],results.Total, results["Total_Correct_Predictions"], "Predictions on Fine-Grained Categories",rot = 45,xlabel = "Category")
        
        return results
        
    
    
    def plot_bar_graph(self,labels, totals, correct_predictions,title,rot = 0,xlabel=""):
        """plots bar graph for different analyses

        Args:
            labels (list): labels to be used for x axis
            totals (list: number of total instances for each class
            correct_predictions (list): number of correctly predicted instances for each class
            title (str): title of graph
            rot (int): rotation for x-axis ticks
            xlabel (str): x label
        """
        plt.bar(labels,totals, color="red",label = "Total",edgecolor='black')
        ax = plt.bar(labels,correct_predictions,color="blue", label ="Correct Predicitons",edgecolor='black')
        plt.legend()
        plt.title(title)
        plt.xticks(ticks = labels, rotation = rot)
        plt.xlabel(xlabel)
        plt.ylabel("Amount")
        plt.show()
        
    def get_examples(self,df,column,results, sort_list= False):
        """pull examples to illustrate specific cases. Adds one text examples from each value present in the specified column to results data

        Args:
            df (df): data to pull from, containing textual data
            column (string): column name to evaluate on
            results (df): df containing information for each classification, (created by all analysis functions)
            sort_list (boolean): if the values need to be sorted for presentation (currently just used by text_length analysis)
        Returns:
            df: updated results 
        """
        column_vals = np.unique(df[column])
        if sort_list:
            column_vals = sorted(column_vals,key=lambda x: float(x.split('-')[0].replace(',','')))
        examples= ["" for x in range(results.shape[0])]
        
        for i in column_vals:
            results_i = results.index[results[column] == i][0] #get the results index
            example = np.random.choice(df[df[column] == i]["Text"], 1)
            examples[results_i] = example[0]
            
        results["Examples"] = examples
        
        return results
        
        
        
        

        
    
    


    

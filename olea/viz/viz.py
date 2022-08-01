import matplotlib.pyplot as plt
import numpy as np


def plot_bar_graph(plot_info, savePlotToFile, title = "", rot = 0, xlabel = ""):
    """Plots bar graph for different analyses

    Args:
        plot info (df): containing the following columns
            Total (list): number of total instances for each category
            Total_Correct_Predictions (list): number of correctly predicted instances for each category
            accuracy (list): accuracy of total_correct_predictions/ total
            savePlotToFile (str): File name for saving plot, empty string will not save a plot
        title (str): title of graph
        rot (int): rotation for x-axis ticks
        xlabel (str): x label
    """
    if plot_info.shape[0] > 2:
        plot_info= plot_info.sort_values(by = "Total",axis = 0,ignore_index =True)
    labels = plot_info.iloc[:,0] 
    totals = plot_info["Total"]
    correct_predictions = plot_info["Total_Correct_Predictions"]
    accuracy = plot_info["Accuracy"]
    
    ha = "center"
    fsize=10
    if len(labels) >2:
        rot = 45
        ha = "right"
        if len(labels) >12:
            fsize= 8
     
        
    fig, ax = plt.subplots()
    ax.bar(labels, totals, color = "red", label = "Total", edgecolor='black')
    ax.bar(labels, correct_predictions, color = "blue", 
                 label = "Correct Predicitons", edgecolor = 'black')
    
    
    # Set an offset that is used to bump the label up a bit above the bar.
    y_offset = 4
    # Add labels to each bar.
    for i, total in enumerate(totals):
      ax.text(totals.index[i], total + y_offset, str(round(accuracy[i]*100,1)) + "%", ha='center')
    
    plt.legend(prop={'size': 8})
    plt.title(title)
    plt.xticks(ticks = np.arange(labels.shape[0]), labels = labels, rotation = rot,ha=ha, rotation_mode='anchor')
    plt.tick_params(axis='x', labelsize=fsize)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Instances")
    
    if savePlotToFile != "":
        plt.savefig(savePlotToFile)
        print("saving figure to " + savePlotToFile)
    plt.show()

def plot_histogram(title = "",hist_bins = 10,legend_location = 'upper right', 
                   xlabel = "", ylabel = "Num. of Instances", 
                   list_of_values = [], 
                   correct_preds = [],
                   accuracy = [],
                   savePlotToFile= ""):
    """Plots histogram specifically for str_len_analysis, but could potentially be used for more

    Args:
        list_of_values (list): number of total instances for each category
        correct_preds (list): number of correctly predicted instances for each category
        accuracy (list): accuracy of total_correct_predictions/ total
        savePlotToFile (str): File name for saving plot, empty string will not save a plot
        title (str): title of graph
        hist_bins (int): number of bins to use
        legend_location (str): legend location
        rot (int): rotation for x-axis ticks
        xlabel (str): x label
        y_label(str):y label
    """
    
    fig, ax = plt.subplots()
    a, bins, _ = ax.hist(list_of_values,bins=hist_bins, color="red", 
                        label = "Total",edgecolor='black')
    _, _, _ = ax.hist(correct_preds,bins = bins, 
                        color="blue", 
                        label = "Correct Prediciton", 
                        edgecolor='black')
    
    # Set an offset that is used to bump the label up a bit above the bar.
    y_offset = 4
    # Add labels to each bar.
    for i, total in enumerate(a):
      ax.text((bins[i] + bins[i+1])/2, total + y_offset, str(round(accuracy[i]*100,1)) + "%", ha='center')
    
    
    plt.legend(loc = legend_location,prop={'size': 8})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(bins)
    
    if savePlotToFile != "":
        plt.savefig(savePlotToFile)
        print("saving figure to " + savePlotToFile)
    plt.show()
    
    

def histogram_values(list_of_values = [], 
                     correct_preds = []) -> list:
    bin_vals, bins, _ = plt.hist(list_of_values, color="red", 
                                 label = "Total",edgecolor='black')
    bin_vals_correct, _, _ = plt.hist(correct_preds,bins = bins, 
                                      color="blue", 
                                      label = "Correct Prediciton", 
                                      edgecolor='black')
    plt.close()
    return bins, bin_vals, bin_vals_correct
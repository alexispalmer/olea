import matplotlib.pyplot as plt
import numpy as np


def plot_bar_graph(labels, totals, correct_predictions,
                   title, rot = 0, xlabel = ""):
    """Plots bar graph for different analyses

    Args:
        labels (list): labels to be used for x axis
        totals (list): number of total instances for each category
        correct_predictions (list): number of correctly predicted instances for each category
        title (str): title of graph
        rot (int): rotation for x-axis ticks
        xlabel (str): x label
    """
    ha = "center"
    fsize=10
    if len(labels) >2:
        rot = 45
        ha = "right"
        if len(labels) >12:
            fsize= 8
        
    plt.bar(labels, totals, color = "red", label = "Total", edgecolor='black')
    ax = plt.bar(labels, correct_predictions, color = "blue", 
                 label = "Correct Predicitons", edgecolor = 'black')
    plt.legend()
    plt.title(title)
    plt.xticks(ticks = np.arange(labels.shape[0]), labels = labels, rotation = rot,ha=ha, rotation_mode='anchor')
    plt.tick_params(axis='x', labelsize=fsize)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Instances")
    plt.show()

def plot_histogram(hist_bins = 10,legend_location = 'upper right', title = "", 
                   xlabel = "", ylabel = "Num. of Instances", 
                   list_of_values = [], 
                   correct_preds = [],
                   ):
    _, bins, _ = plt.hist(list_of_values,bins=hist_bins, color="red", 
                        label = "Total",edgecolor='black')
    _, _, _ = plt.hist(correct_preds,bins = bins, 
                        color="blue", 
                        label = "Correct Prediciton", 
                        edgecolor='black')
    plt.legend(loc = legend_location)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(bins)
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
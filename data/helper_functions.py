from collections import Counter, defaultdict
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score


def get_num_rows_columns(dataframe):
    """calculates the number of rows and columns in a dataframe

    Args:
        dataframe (pandas.DataFrame): a pandas dataframe

    Returns:
        num_rows (int): number of rows in the dataframe
        num_cols (int): number of columns in the dataframe
    """    
    num_columns = len(list(dataframe.columns))
    num_rows = int(num_columns ** 0.5) + 1
    num_cols = num_columns // num_rows + 1
    return num_rows, num_cols


def count_IDs(dataframes):
    
    """counts the number of IDs in each dataset and the total number of IDs in all datasets
    
    Args:
        dataframes (dict): dictionary of dataframes, where the key is the name of the dataset and the value is the dataframe itself

    Returns:
        all_IDed_neurons (dict): dictionary of all neurons and their counts
        IDs_per_set (dict): dictionary of the number of IDs per dataset
    """    
    
    # initialising the dictionaries for counting the IDs
    all_IDed_neurons = Counter()
    IDs_per_set = defaultdict()


    for key, value in dataframes.items():
        # take only columns that have IDs (e.g. "AVAR","RIBL",..)
        IDed_neurons = [column for column in value.columns if "neuron" not in column]
        dataframes[key] = value[IDed_neurons]

        #some cleaning, removing question marks (rebecca's data), useless columns and duplicated columns 
        dataframes[key].columns = dataframes[key].columns.str.replace('?', '', regex=True)
        dataframes[key] = dataframes[key].loc[:, ~dataframes[key].columns.duplicated()] 
        dataframes[key] = dataframes[key].drop(columns=[columnname for columnname in ['is this OLQ or URA','OLQDLorR','masked','retrace',''] if columnname in dataframes[key].columns])
        
        # incrementing the counter for each ID
        for ID in list(dataframes[key].columns):
            all_IDed_neurons[ID] += 1

        # counting the number of IDs per dataset
        IDs_per_set[key] = len(dataframes[key].columns)
        
    return all_IDed_neurons, IDs_per_set


def visualize_IDs(dictionary, title, xlabel, ylabel, coloring="tab:orange", display_all_values=False):
    
    """plots a dictionary of neurons and their values (e.g. counts) as a bar chart

    Args:
        dictionary: dictionary of neurons and their values (e.g. counts)
        title: title of the plot
        xlabel: label of the x-axis
        ylabel: label of the y-axis
        coloring: color of the bars
        display_all_values: if True, all values are displayed on the y-axis, if False, only every 10th value is displayed
        
    Returns:
        fig, ax: figure and axis of the plot
    """    
    
    
    dictionary = dict(sorted(dictionary.items(), key=lambda item: item[1], reverse=False))
    dict_keys = list(dictionary.keys())
    dict_values = list(dictionary.values())

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(15, 7))  # You can adjust the width as needed
    plt.ylim(min(dict_values)-(max(dict_values)*0.05), max(dict_values)+(max(dict_values)*0.05))
    
    # Create the cumulative bar chart and add markers on top of each bar
    ax.bar(dict_keys, dict_values,
        color=coloring, alpha=0.7, width=0.5)
    ax.plot(dict_keys, dict_values, marker='o',
            color=coloring, linestyle='', label='Markers')


    x_positions = [neurons_key-0.1 for neurons_key in range(len(dict_keys))]
    # Set y-axis and x-axis labels
    if display_all_values:
        ax.set_yticks(dict_values)
    else:
        step = len(dict_values) // 10
        # Use slicing to get 10 equidistant values from the list of y-values
        ax.set_yticks(dict_values[::step])
    ax.set_xticks(x_positions)
    # Adjust rotation and alignment as needed
    ax.set_xticklabels(dict_keys, rotation=90)

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax



def get_R2_predictions(dataframes, all_IDed_neurons):
    
    """calculates the R2 scores of each neuron and the predictions of each neuron

    Args:
        dataframes (dict): dictionary of dataframes, where the key is the name of the dataset and the value is the dataframe itself
        all_IDed_neurons (dict): dictionary of all neurons and their counts

    Returns:
        avg_r2 (dict): dictionary of neurons and their average R2 scores
        predictions (dict): dictionary of neurons and their predicted activity
        raw_data (dict): dictionary of neurons and their raw data
    """    
    
    rsquareds = defaultdict()
    raw_data = defaultdict(lambda: defaultdict(list))
    predictions = defaultdict(lambda: defaultdict(list))
    threshold = 10
    avg_r2 = defaultdict()

    cv = KFold(n_splits=5, random_state=1, shuffle=True) 

    for key, dataframe in dataframes.items():

        for neuron in dataframe.columns:

            # skip neurons that have been IDed less than 10 times
            if all_IDed_neurons[neuron] < threshold:
                continue

            # y is our response variable, X is our explanatory variable
            y = dataframe[neuron]
            X = dataframe.drop(columns=[neuron])

            # train a simple linear regression model and get the r^2 score and prediction of the neuron
            model = LinearRegression().fit(X, y)

            # quantify how good the model is by looking at the R2 values in a cross-validated fashion
            scores = cross_val_score(model, X, y, scoring='r2',
                            cv=cv, n_jobs=-1)

            rsquare = np.mean(scores)

            if neuron in rsquareds:
                rsquareds[neuron] += rsquare
            else:
                rsquareds[neuron] = rsquare

            # store the prediction of the neuron in a dictionary
            prediction = model.predict(X)
            predictions[neuron][key] = prediction
            raw_data[neuron][key] = np.array(y)


    # averaging the R2 scores over all datasets
    for neuron in rsquareds:
        avg_r2[neuron] = rsquareds[neuron]/all_IDed_neurons[neuron]
        
    return avg_r2, predictions, raw_data
        
        
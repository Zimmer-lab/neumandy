import os
import sys
current_directory = os.getcwd()  # NOQA
parent_directory = os.path.join(current_directory, '..')  # NOQA
sys.path.append(parent_directory)  # NOQA

import importlib
import dill
from sklearn.decomposition import FastICA, PCA
from wbstruct_converter import utils
import wbstruct_converter.utils.wbstruct_dicts_to_dataframes as wbstruct_dataframes
import textwrap
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter, defaultdict
from sklearn.model_selection import LeaveOneOut
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


# for loading the pkl objects we need to import the module
sys.modules['utils'] = utils

sys.path.append('C:\\Users\\LAK\\Documents\\lianaforks\\dev\\wbfm')

import wbfm.utils.general.utils_behavior_annotation as behavior_annotation  # NOQA
import wbfm.utils.general.postprocessing.utils_imputation as utils_imputation  # NOQA
import wbfm.utils.visualization.plot_traces as plot_traces  # NOQA
import wbfm.utils.visualization.utils_plot_traces as utils_plot_traces  # NOQA

# for wrapping outputs
wrapper = textwrap.TextWrapper(width=50)


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
        IDed_neurons = [
            column for column in value.columns if "neuron" not in column]
        dataframes[key] = value[IDed_neurons]

        # some cleaning, removing question marks (rebecca's data), useless columns and duplicated columns
        dataframes[key].columns = dataframes[key].columns.str.replace(
            '?', '', regex=True)
        dataframes[key] = dataframes[key].loc[:, ~
                                              dataframes[key].columns.duplicated()]
        dataframes[key] = dataframes[key].drop(columns=[columnname for columnname in [
                                               'is this OLQ or URA', 'OLQDLorR', 'masked', 'retrace', ''] if columnname in dataframes[key].columns])

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

    dictionary = dict(
        sorted(dictionary.items(), key=lambda item: item[1], reverse=False))
    dict_keys = list(dictionary.keys())
    dict_values = list(dictionary.values())

    # Create the figure and axis
    # You can adjust the width as needed
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.ylim(min(dict_values)-(max(dict_values)*0.05),
             max(dict_values)+(max(dict_values)*0.05))

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


def plot_from_stacked_imputed(length_dict, stacked_dataframe, imputed_dataframe, saving_path):
    """plots the stacked and imputed dataframes and saves the plots

    Args:
        length_dict (dict): dictionary of the number of observations per dataset
        stacked_dataframe (pd.DataFrame): dataframe of the stacked data
        imputed_dataframe (pd.DataFrame): dataframe of the imputed data
        saving_path (str): path to save the plots
    """

    start_index = 0
    count = 0

    # we will unstack the dataframe and plot the traces for each dataset
    for obs_count in list(length_dict.values()):

        # we take the number of observations from the length dictionary and add it to the start index
        end_index = start_index + obs_count
        df_imputed = imputed_dataframe.iloc[start_index:end_index]
        df_unimputed = stacked_dataframe.iloc[start_index:end_index]

        # 2 dataframe grid plots, imputed in blue (first argument, such that it is in the back) and unimputed in orange (second argument, on top)
        fig = plot_traces.make_grid_plot_from_two_dataframes(
            df_imputed, df_unimputed)
        # fig, ax = plot_traces.make_grid_plot_from_dataframe(df_imputed)

        # save all plots in a folder
        pathname = saving_path + list(length_dict.keys())[count] + ".png"
        fig.savefig(pathname)
        plt.close(fig)
        start_index = end_index
        count += 1


def plot_from_single_imputed(raw_data, predictions, delta_path, model_path, plot_kwargs):
    """plots the raw data against the predictions and the delta between the two and saves the plots

    Args:
        raw_data (defaultdict): dataframe of the raw data
        predictions (defaultdict): dictionary of neurons and their predictions
        delta_path (str): path to save the delta plots
        model_path (str): path to save the model plots
        **plot_kwargs: additional arguments for the plot
    """

    modelled_activity_patterns = defaultdict()

    for neuron, df in predictions.items():
        raw_neuron = list(raw_data[neuron].values())
        modelled_neuron = list(df.values())
        diff = [raw_neuron[i]-modelled_neuron[i]
                for i in range(len(raw_neuron))]

        modelled_activity_patterns[neuron] = pd.DataFrame(modelled_neuron).T

        # Calculate the number of rows and columns for subplots
        num_rows, num_cols = get_num_rows_columns(
            modelled_activity_patterns[neuron])

        figsample, ax = plt.subplots(num_rows, num_cols, figsize=(12, 8))

        # create delta figures
        fig_delta, ax = plot_traces.make_grid_plot_from_dataframe(
            pd.DataFrame(diff).T, fig=figsample)
        fig_delta.savefig(delta_path+neuron+".png")

        # clear figure
        plt.cla()

        figsample, ax = plt.subplots(num_rows, num_cols, figsize=(12, 8))

        fig, ax = plot_traces.make_grid_plot_from_dataframe(
            modelled_activity_patterns[neuron], fig=figsample)

        fig, ax = plot_traces.make_grid_plot_from_dataframe(pd.DataFrame(
            raw_neuron).T, fig=fig, twinx_when_reusing_figure=True, **plot_kwargs)

        # save all plots in a folder
        fig.savefig(model_path+neuron+".png")


def find_percent(data, min_value):
    """calculates the percentile of a value in a list

    Args:
        data (list): list of values
        min_value (float): value for which the percentile is calculated

    Returns:
        percent: percentile of the value
    """
    sorted_data = sorted(data)
    rank = sorted_data.index(min_value) + 1
    total_points = len(sorted_data)
    percent = 100 - ((rank - 0.5) / total_points) * 100
    return percent


def get_behavioural_states(dataframe):
    """taken from wbfm function @approximate_turn_annotations_using_ids and modified to return the behavioural states from a dataframe and not from a project

    Args:
        dataframe (pd.DataFrame): dataframe of the data

    Returns:
        turn_vec (pd.Series): a series of the behavioural states
    """

    y_dorsal = behavior_annotation.combine_pair_of_ided_neurons(
        dataframe, base_name='SMDD')
    y_ventral = behavior_annotation.combine_pair_of_ided_neurons(
        dataframe, base_name='SMDV')
    y_reversal = behavior_annotation.combine_pair_of_ided_neurons(
        dataframe, base_name='AVA')

    dorsal_vec = behavior_annotation.calculate_rise_high_fall_low(y_dorsal)
    ventral_vec = behavior_annotation.calculate_rise_high_fall_low(
        y_ventral)
    reversal_vec = behavior_annotation.calculate_rise_high_fall_low(
        y_reversal)

    ava_fall_starts, ava_fall_ends = behavior_annotation.get_contiguous_blocks_from_column(
        reversal_vec == 'fall', already_boolean=True)
    ava_high_starts, ava_high_ends = behavior_annotation.get_contiguous_blocks_from_column(
        reversal_vec == 'high', already_boolean=True)
    ava_rise_starts, ava_rise_ends = behavior_annotation.get_contiguous_blocks_from_column(
        reversal_vec == 'rise', already_boolean=True)

    turn_vec = pd.Series(np.zeros_like(reversal_vec),
                         index=reversal_vec.index, dtype=object)
    for s, e in zip(ava_fall_starts, ava_fall_ends):
        if s <= 1:
            continue
        # Check if dorsal or ventral are in a rise state, including some time after
        e_padding = e + 10
        len_dorsal_rise = len(np.where(dorsal_vec[s:e_padding] == 'rise')[0])
        len_ventral_rise = len(np.where(ventral_vec[s:e_padding] == 'rise')[0])

        if len_ventral_rise > len_dorsal_rise:

            turn_vec[s:e] = 'ventral'
        elif len_ventral_rise < len_dorsal_rise:
            turn_vec[s:e] = 'dorsal'
        elif len_ventral_rise == 0 and len_dorsal_rise == 0:
            continue
        else:
            # This means they were both rising the same non-zero amount
            if np.mean(y_ventral[s:e_padding]) > np.mean(y_dorsal[s:e_padding]):
                turn_vec[s:e] = 'ventral'
            else:
                turn_vec[s:e] = 'dorsal'

    for s, e in zip(ava_high_starts, ava_high_ends):
        turn_vec[s:e] = 'reversal'

    for s, e in zip(ava_rise_starts, ava_rise_ends):
        turn_vec[s:e] = 'reversal'

    turn_vec.replace(0, 'forward', inplace=True)

    return turn_vec


def get_LLO_PCAs(dataframe, n_components=3):
    """calculates the PCA loadings for each neuron using leave one out cross validation

    Args:
        dataframe (pd.DataFrame): dataframe of the data
        n_components (int, optional): number of PCA components. Defaults to 3.

    Returns:
        pca_all_splits (defaultdict): dictionary of the PCA loadings for each neuron
    """

    loo = LeaveOneOut()
    pca_all_splits = defaultdict(list)

    # 73 iterations are done because we have 73 neurons
    for train_index, test_index in loo.split(dataframe):
        X_train = dataframe.iloc[train_index]

        # Fit the PCA model on the training data
        pca = PCA(n_components=n_components)
        pca_neuron_loo = pca.fit_transform(X_train)

        # Retrieve and store the PCA loadings of the first component as a DataFrame
        for i in range(n_components):
            variable_name = f"pca{i+1}_all_splits"
            pca_df_loo = pd.DataFrame(pca_neuron_loo[:, i])
            pca_df_loo["neuron"] = X_train.index
            pca_df_loo = pca_df_loo.rename(columns={0: 'Mode {}'.format(i+1)})
            pca_all_splits[variable_name].append(pca_df_loo)

    return pca_all_splits


def get_mahalanobis_distances(dataframe):
    # computing the covariance which is important for the mahalanobis distance

    cov_matrix = dataframe.cov()
    # compute inverse of covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mean_predictors = np.mean(dataframe, axis=0)
    mahalanobis_distances = [mahalanobis(obs[1], mean_predictors, inv_cov_matrix) for obs in dataframe
                             .iterrows()]
    return mahalanobis_distances


def plot_PCs(dataframe, turn_vec, filename):
    plotly_pca, names = utils_plot_traces.modify_dataframe_to_allow_gaps_for_plotly(
        dataframe, [0, 1, 2], 'state')
    state_codes = turn_vec.unique()
    phase_plot_list = []
    for i, state_code in enumerate(state_codes):
        phase_plot_list.append(
            go.Scatter3d(x=plotly_pca[names[0][i]], y=plotly_pca[names[1][i]], z=plotly_pca[names[2][i]], mode="lines",
                         name=state_code))

    fig = go.Figure()
    fig.add_traces(phase_plot_list)
    fig.update_layout(scene=dict(
        xaxis_title='Mode 1',
        yaxis_title='Mode 2',
        zaxis_title='Mode 3'))
    fig.write_html(filename)
    fig.show()

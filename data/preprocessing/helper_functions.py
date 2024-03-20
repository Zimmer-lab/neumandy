import os
import sys
import imageio
from matplotlib.ticker import MaxNLocator
import pynumdiff as pdiff
current_directory = os.getcwd()  # NOQA
parent_directory = os.path.join(current_directory, '..')  # NOQA
sys.path.append(parent_directory)  # NOQA
from dash.dependencies import Input, Output
from sklearn.decomposition import PCA
import textwrap
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter, defaultdict
from sklearn.model_selection import LeaveOneOut
import plotly.graph_objects as go
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
import dash
from dash import dcc, html


import wbfm.utils.general.utils_behavior_annotation as behavior_annotation  # NOQA
import wbfm.utils.visualization.plot_traces as plot_traces  # NOQA
import wbfm.utils.visualization.utils_plot_traces as utils_plot_traces  # NOQA

# for wrapping outputs
wrapper = textwrap.TextWrapper(width=50)


def interpolate(vector, indices):
    """interpolates a vector to a certain length

    Args:
        vector (list): list of values
        indices (list): list of indices

    Returns:
        vector: interpolated vector
    """
    vector = np.interp(indices, np.linspace(0, 1, len(vector)), vector)
    return vector


def resample(dataframe, lengths, frames_num=3529):
    """resamples the data to the same length

    Args:
        dataframe (): dataframe of the stacked data
        length_dict (): dictionary of the number of observations per dataset
        frames_num (int, optional): _description_. Defaults to 3529.

    Returns:
        _type_: _description_
    """

    # we will unstack the dataframe and plot the traces for each dataset
    start_index = 0
    resampled_dataframes = []
    final_indexes = []

    for obs_count in lengths:

        # we take the number of observations from the length dictionary and add it to the start index
        end_index = start_index + obs_count
        df = dataframe.iloc[start_index:end_index].copy()
        if obs_count < frames_num:
            if "state" in df.columns:
                # Interpolate the values between the first and last elements
                # index of current df
                indices = np.linspace(0, frames_num, obs_count)
                df.index = indices

                diff = frames_num - obs_count
                nan_data = pd.DataFrame(
                    np.nan, index=range(diff), columns=df.columns)
                new_indices = np.linspace(1, frames_num-1, diff)
                nan_data.index = new_indices
                df = pd.concat([df, nan_data])
                df = df.sort_index().reset_index(drop=True)
                interpolated_df = df.loc[:, ~df.columns.isin(["state", "dataset"])].interpolate(
                    method="linear", axis=0)
                interpolated_df["state"] = df["state"].interpolate(
                    method="backfill", limit_direction="backward")
                if "dataset" in df.columns:
                    interpolated_df["dataset"] = df["dataset"].interpolate(
                        method="backfill", limit_direction="backward")
                df = interpolated_df
        if obs_count > frames_num:
            # index of current df
            indices = np.linspace(0, obs_count-1, frames_num, dtype=int)
            df = df.iloc[indices]

        resampled_dataframes.append(df)

        start_index = end_index

    resampled_dataframe = pd.concat(
        resampled_dataframes, axis=0, ignore_index=True)

    return resampled_dataframe


def truncate(dataframe, n=100):
    """truncates the first and the last n frames of the data

    Args:
        dataframe (pd.DataFrame): dataframe of the stacked data
        n (int, optional): _description_. Defaults to 100.

    Returns:
        truncated_dataframe (pd.DataFrame): truncated dataframe
    """

    # we will unstack the dataframe and plot the traces for each dataset
    start_index = 0
    truncated_dataframes = []

    for obs_count in dataframe.groupby("dataset").size().values:

        end_index = start_index + obs_count
        df = dataframe.iloc[start_index+n:end_index-n]

        # we replace the dataframe with the interpolated dataframe
        truncated_dataframes.append(df)

        start_index = end_index

    truncated_dataframe = pd.concat(
        truncated_dataframes, axis=0, ignore_index=True)

    return truncated_dataframe


def compute_derivatives(dataframe, length_dict, iterations=1, gamma=0.01, dt=1/3):
    """computes the derivatives of the data

    Args:
        dataframe (pd.DataFrame): dataframe of the data

    Returns:
        dataframe (pd.DataFrame): dataframe of the data with derivatives
    """

    resampled_derivatives = dataframe.copy()

    start_index = 0
    # we compute the derivatives of the data
    for obs_count in length_dict.values():
        end_index = start_index + obs_count
        for col_index in range(len(dataframe.columns)):
            # x_hat: estimated (smoothed) x, dxdt_hat: estimated dx/dt, [1, 0.0001]: regularization parameters -> gamma=0.2 is too high, derivatives become too blocky
            x_hat, dxdt_hat = pdiff.total_variation_regularization.iterative_velocity(
                resampled_derivatives.iloc[start_index:end_index, col_index], dt, [iterations, gamma])
            resampled_derivatives.iloc[start_index:end_index,
                                       col_index] = dxdt_hat
        start_index = end_index

    return resampled_derivatives


def compute_cumsum(dataframe, length_dict):
    start_index = 0
    for obs_count in length_dict.values():
        end_index = start_index + obs_count
        for col_index in range(len(dataframe.columns)):
            integrated = np.cumsum(
                dataframe.iloc[start_index:end_index, col_index])
            dataframe.iloc[start_index:end_index, col_index] = integrated + \
                abs(integrated.min()) + 0.01
        start_index = end_index


def normalize_per_dataset(dataframe, lengths, scaler):
    """normalizes the data per dataset

    Args:
        dataframe (pd.DataFrame): dataframe of the stacked data
        lengths (list): list of the number of observations per dataset

    Returns:
        normalized_dataframe: normalized dataframe
    """

    normalized_dfs = []

    start_index = 0
    # we will unstack the dataframe and plot the traces for each dataset
    for obs_count in lengths:

        # we take the number of observations from the length dictionary and add it to the start index
        end_index = start_index + obs_count
        resampled_dataframe_df = dataframe.iloc[start_index:end_index]

        normalized_dfs.append(pd.DataFrame(scaler.fit_transform(
            resampled_dataframe_df), columns=resampled_dataframe_df.columns))

        start_index = end_index

    normalized_dataframe = pd.concat(normalized_dfs, ignore_index=True)
    return normalized_dataframe


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


def visualize_fps(dataframe, title, xlabel, ylabel, coloring="tab:red", display_all_values=False):
    """plots a dictionary of neurons and their values (e.g. counts) as a bar chart

    Args:
        dataframe: dataframe with column 'dataset' to indicate how many time points belong to the recording
        title: title of the plot
        xlabel: label of the x-axis
        ylabel: label of the y-axis
        coloring: color of the bars
        display_all_values: if True, all values are displayed on the y-axis, if False, only every 10th value is displayed

    Returns:
        fig, ax: figure and axis of the plot
    """

    dict_keys = list(dataframe["dataset"].unique())
    dict_values = list(dataframe.groupby('dataset').size().values)

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

    plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))

    ax.set_xticks(x_positions)
    # Adjust rotation and alignment as needed
    ax.set_xticklabels(dict_keys, rotation=90)

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def vip(model):
    """calculates the VIP scores of a PLSR model as described on github in a scikit-learn thread: https://github.com/scikit-learn/scikit-learn/issues/7050#issuecomment-345208503

    Args:
        model (PLSRegression): PLSR model

    Returns:
        list: VIP scores
    """

    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array(
            [(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
    return vips


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
    top_predictors = defaultdict(lambda: defaultdict(list))
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
            N = 3  # number of components to use for PLSR

            # train a simple linear regression model and get the r^2 score and prediction of the neuron
            model = PLSRegression(n_components=N).fit(X, y)

            # quantify how good the model is by looking at the R2 values in a cross-validated fashion
            r2s = cross_val_score(model, X, y, scoring='r2',
                                  cv=cv, n_jobs=-1)
            rsquare = np.mean(r2s)

            if neuron in rsquareds:
                rsquareds[neuron] += rsquare
            else:
                rsquareds[neuron] = rsquare

            # store the prediction of the neuron in a dictionary
            prediction = model.predict(X)
            predictions[neuron][key] = prediction

            # Calculate VIP scores
            VIP_scores = vip(model)
            predictors = [(list(X.columns)[i], VIP_scores[i])
                          for i in VIP_scores.argsort()[::-1]]

            top_VIPs = VIP_scores.argsort()[::-1][:5]
            top_5_predictors = [(list(X.columns)[i], VIP_scores[i])
                                for i in top_VIPs]
            for i in range(len(predictors)):
                predictor = predictors[i][0]
                if predictor in top_predictors[neuron]:
                    top_predictors[neuron][predictor].append(
                        predictors[i][1])
                else:
                    top_predictors[neuron][predictor] = [
                        predictors[i][1]]

            # add the raw data to a dictionary
            raw_data[neuron][key] = np.array(y)

    # averaging the R2 scores over all datasets
    for neuron in rsquareds:
        avg_r2[neuron] = rsquareds[neuron]/all_IDed_neurons[neuron]

    return avg_r2, predictions, top_predictors, raw_data


def plot_from_stacked_imputed(dataset_dict, df1, df2, saving_path):
    """plots the stacked and imputed dataframes and saves the plots

    Args:
        dataset_dict (dict): dictionary of the number of observations per dataset
        dataframe1 (pd.DataFrame): dataframe of the stacked data
        dataframe2 (pd.DataFrame): dataframe of the imputed data
        saving_path (str): path to save the plots
    """

    start_index = 0
    count = 0

    dataframe1 = df1.copy()
    dataframe2 = df2.copy()

    # we will unstack the dataframe and plot the traces for each dataset
    for obs_count in dataset_dict.values():

        if "state" in dataframe1.columns:
            dataframe1.drop(columns=["state"], inplace=True)
        if "state" in dataframe2.columns:
            dataframe2.drop(columns=["state"], inplace=True)

        # we take the number of observations from the length dictionary and add it to the start index
        end_index = start_index + obs_count
        df_dataframe2 = dataframe2.iloc[start_index:end_index]
        df_dataframe1 = dataframe1.iloc[start_index:end_index]

        # 2 dataframe grid plots, imputed in blue (first argument, such that it is in the back) and unimputed in orange (second argument, on top)
        fig = plot_traces.make_grid_plot_from_two_dataframes(
            df_dataframe1, df_dataframe2, twinx_when_reusing_figure=True)
        # fig, ax = plot_traces.make_grid_plot_from_dataframe(df_imputed)

        # save all plots in a folder
        pathname = saving_path + list(dataset_dict.keys())[count] + ".png"
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


def determine_turn(dataframe, original_turn_vec):
    in_turn = False
    count = 0
    smdv = 0
    smdd = 0
    actual_turn_vec = original_turn_vec.copy()
    for idx, state in enumerate(original_turn_vec):
        if state == "turn":
            in_turn = True
            count = count + 1

            smdvr = dataframe.loc[idx, "SMDVR"]
            smdvl = dataframe.loc[idx, "SMDVL"]
            smdv = smdv + (smdvr + smdvl) / 2

            smddr = dataframe.loc[idx, "SMDDR"]
            smddl = dataframe.loc[idx, "SMDDL"]
            smdd = smdd + (smddr + smddl) / 2

        else:
            if in_turn:
                if (smdv/count) > (smdd/count):
                    actual_turn_vec[idx-count:idx] = "ventral"
                else:
                    actual_turn_vec[idx-count:idx] = "dorsal"
                smdv = 0
                smdd = 0
                count = 0
                in_turn = False
            actual_turn_vec[idx] = state
    return actual_turn_vec


def apply_PCA_with_smoothing(dataframe):
    pca = PCA(n_components=3)
    dataframe_pca = pd.DataFrame(pca.fit_transform(
        dataframe.loc[:, ~dataframe.columns.isin(["state", "dataset"])]))
    window_size = 10
    # Applying a 10-sample sliding average for smoother visualizations!
    for i in range(3):
        dataframe_pca[i] = np.convolve(dataframe_pca[i], np.ones(
            window_size)/window_size, mode='same')

    return dataframe_pca


def plot_PCs(dataframe, filename='PCA_plot.html', variances=None):
    """plots the first three principal components of the data

    Parameters
    ----------
        dataframe (pd.DataFrame): dataframe of the data with a column of behavioural states
        filename (str): filename of the plot
        variances (list): list of the variances explained by each principal component

    Returns
    ----------
        fig (go.Figure()): figure of the plot
    """

    plotly_pca, names = utils_plot_traces.modify_dataframe_to_allow_gaps_for_plotly(
        dataframe, [0, 1, 2], 'state')
    state_codes = dataframe['state'].unique()
    phase_plot_list = []
    custom_colors = {
        'reversal': 'rgb(255,99,71)',
        'forward': 'rgb(100,149,237)',
        'dorsal': 'rgb(154,205,50)',
        'ventral': 'rgb(255,215,0)',
        'sustained reversal': 'rgb(128, 0, 32)',
        'post reversal': 'rgb(130, 30, 20)'
    }

    for i, state_code in enumerate(state_codes):
        phase_plot_list.append(
            go.Scatter3d(x=plotly_pca[names[0][i]], y=plotly_pca[names[1][i]], z=plotly_pca[names[2][i]], mode="lines",
                         name=state_code, line=dict(color=custom_colors[state_code], width=3)))

    fig = go.Figure()
    fig.add_traces(phase_plot_list)
    if variances is not None:
        scene = dict(xaxis_title=f"PC 1 ({variances[0]:.2f}%)",
                     yaxis_title=f"PC 2 ({variances[1]:.2f}%)",
                     zaxis_title=f"PC 3 ({variances[2]:.2f}%)")
    else:
        scene = dict(xaxis_title="PC 1",
                     yaxis_title="PC 2",
                     zaxis_title="PC 3")

    fig.update_layout(scene=scene)
    fig.write_html(filename)
    # fig.show()
    return fig


def plot_PCs_separately(datasets):
    # datasets is a dictionary of dataframe containing the data projected to PC space

    app = dash.Dash(__name__)

    @app.callback(
        Output('graph', 'figure'),
        [Input('slider', 'value')])
    def update_graph(selected_dataset):
        keyname = list(datasets.keys())[selected_dataset]
        fig = plot_PCs(datasets[keyname])
        return fig

    app.layout = html.Div([
        dcc.Graph(id='graph'),
        dcc.Slider(
            id='slider',
            min=1,
            max=len(datasets.keys()),
            value=1,
            step=1
        )
    ])

    return app


def plot_PCs_iteratively(datasets):
    # datasets is a dictionary of dataframe containing the data projected to PC space

    app = dash.Dash(__name__)
    #
    #
    # fig =

    @app.callback(
        Output('graph', 'figure'),
        [Input('slider', 'value')])
    def update_graph(selected_dataset):

        if selected_dataset == 1:
            keyname = list(datasets.keys())[0]
            df = datasets[keyname]
            return plot_PCs(df)

        else:
            selected_datasets = []
            for i in range(selected_dataset):
                keyname = list(datasets.keys())[i]
                selected_datasets.append(datasets[keyname])

            df = pd.concat(selected_datasets, ignore_index=True)
            fig = plot_PCs(df)
            return fig

    app.layout = html.Div([
        dcc.Graph(id='graph'),
        dcc.Slider(
            id='slider',
            min=1,
            max=len(datasets.keys()),
            value=1,
            step=1
        )
    ])

    return app


def plot_PC_gif(dataframe, turn_vec, fn):
    plotly_pca, names = utils_plot_traces.modify_dataframe_to_allow_gaps_for_plotly(
        dataframe, [0, 1, 2], 'state')
    state_codes = turn_vec.unique()

    custom_colors = {
        'reversal': 'rgb(255,99,71)',
        'forward': 'rgb(100,149,237)',
        'dorsal': 'rgb(154,205,50)',
        'ventral': 'rgb(255,215,0)',
        'sustained reversal': 'rgb(128, 0, 32)'
    }
    phase_plot_list = []
    for i, state_code in enumerate(state_codes):
        phase_plot_list.append(
            go.Scatter3d(x=plotly_pca[names[0][i]], y=plotly_pca[names[1][i]], z=plotly_pca[names[2][i]], mode='lines',
                         name=state_code, line=dict(color=custom_colors[state_code], width=3)))

    fig = go.Figure()
    fig.add_traces(phase_plot_list)
    fig.update_layout(scene=dict(camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)), xaxis_title='Mode 1',
                                 yaxis_title='Mode 2',
                                 zaxis_title='Mode 3'))

    num_frames = 100  # Adjust the number of frames as needed

    rotation_angles = np.linspace(0, 2 * np.pi, num_frames)

    os.makedirs("frames", exist_ok=True)
    for i, angle in enumerate(rotation_angles):
        fig.update_layout(scene_camera_eye=dict(
            x=np.cos(angle) * 1.25, y=np.sin(angle) * 1.25, z=1.25))
        image_filename = os.path.join("frames", f"frame_{i:03d}.png")

        # note: this requires a downgrade of the engine kaleido from 0.2.1 to 0.1.0 - pathetic, I know
        fig.write_image(image_filename)

    # create GIF out of all the different angles of the principal components

    images = []

    for filename in os.listdir("frames"):
        if filename.endswith(".png"):
            images.append(imageio.imread(os.path.join("frames", filename)))

    imageio.mimsave(fn, images, duration=0.1)


def apply_isolation_forest(X, contamination=0.025):
    if "dataset" in X.columns:
        X = X.drop(columns=["dataset", "state"])
    clf = IsolationForest(n_estimators=100, max_samples='auto',
                          contamination=contamination, warm_start=True)
    clf.fit(X)  # fit 10 trees
    X["outlier"] = ["outlier" if x == -
                    1 else "no outlier" for x in clf.predict(X).tolist()]
    return X

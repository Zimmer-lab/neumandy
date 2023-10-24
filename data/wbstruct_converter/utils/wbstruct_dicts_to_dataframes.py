
import numpy as np
import pandas as pd
import copy
import dill
import os


def saving_as_hdf5(dataframes):
    """This function saves the dataframes as hdf5 files"""

    output_directory = "hdf5_files"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for trial, trialvalue in dataframes.items():
        filename = os.path.join(output_directory, f"{trial}.h5")
        trialvalue.to_hdf(filename, key='data', mode='w')

    print("Dataframes stored as hdf5 files in the directory 'hdf5_files'")

    return None


def saving_as_pkl(dataframes):
    with open('dataframes.pkl', 'wb') as file:
        dill.dump(dataframes, file)
    
    print("Dataframes stored as a .pkl file. Refer to documentation to load the file.")

    return None


def loading_pkl(dataframes):
    with open(dataframes ,'rb') as f:
        return dill.load(f, ignore=False)



def get_dataframes(dictionaries, recording_type, save_as_hdf5=False):
    """This function converts the dictionary of wbstruct data into a dictionary of dataframes

    Args:
        dictionaries (defaultdict): a dictionary of dictionaries that contains the data from all matlab files
        recording_type (string): the type of imaging values we want, e.g. deltaFOverF or deltaFoverF_bc 

    Returns:
        defaultdict: a dictionary of dataframes that contains the data from all matlab files
    """

    datasets = copy.deepcopy(dictionaries)

    for trial, trialvalue in dictionaries.items():

        # merging head and tail data

        merged_datasets = np.hstack(
            (trialvalue["Head"][recording_type], trialvalue["Tail"][recording_type]))

        id_length = merged_datasets.shape[1]
        id_names = trialvalue["Head"]["ID1"] + trialvalue["Tail"]["ID1"]
        colnames = [f"neuron_{i:03d}" for i in range(id_length)]
        colnames = [dummy if pd.isna(
            ID) else ID for dummy, ID in zip(colnames, id_names)]
        datasets[trial] = pd.DataFrame(merged_datasets, columns=colnames)

    if save_as_hdf5:
        saving_as_hdf5(datasets)
    else:
        saving_as_pkl(datasets)

    return datasets


def get_dataframes_from_excel(dictionaries, IDs, recording_type, save_as_hdf5=False):
    """This function converts the dictionary of wbstruct data into a dictionary of dataframes and 
        uses the IDs from a separate dictionary

    Args:
        dictionaries (defaultdict): a dictionary of dictionaries that contains the data from all matlab files
        IDs (defaultdict): a dictionary that contains the IDs from every recording
        recording_type (string): the type of imaging values we want, e.g. deltaFOverF or deltaFoverF_bc 

    Returns:
        defaultdict: a dictionary of dataframes that contains the data from all matlab files
    """

    dictofIDs = copy.deepcopy(IDs)
    datasets = copy.deepcopy(dictionaries)


    for trial, trialvalue in dictionaries.items():

        if "notUsed" not in trial:

            id_names = dictofIDs[trial]

            # merging head and tail data if both are available

            try:
                merged_datasets = np.hstack(
                    (trialvalue["Head"][recording_type], trialvalue["Tail"][recording_type]))

            except ValueError as e:
                merged_datasets = np.vstack(
                    (trialvalue["Head"][recording_type]))

            id_length = merged_datasets.shape[1]

            # if the number of neurons in the recording is not equal to the number of IDs we want to exclude this recording

            if not len(id_names) == id_length:
                if not len(id_names) == (id_length-trialvalue["Head"][recording_type].shape[1]):
                    del datasets[trial]
                    continue
                else:
                    id_length = id_length - \
                        trialvalue["Head"][recording_type].shape[1]
                    merged_datasets = trialvalue["Tail"][recording_type]

            # creating column names for the DF from the IDs that we got
            colnames = [f"neuron_{i:03d}" for i in range(id_length)]
            colnames = [dummy if pd.isna(
                ID) else ID for dummy, ID in zip(colnames, id_names)]

            datasets[trial] = pd.DataFrame(
                merged_datasets, columns=colnames)

    if save_as_hdf5:
        saving_as_hdf5(datasets)
    else:
        saving_as_pkl(datasets)

    return datasets


import numpy as np
import pandas as pd
import copy


def get_dataframes(dictionaries, recording_type):

    datasets = copy.deepcopy(dictionaries)

    for trial, trialvalue in dictionaries.items():

        id_names = dictionaries[trial]["Head"]["ID1"] + \
            dictionaries[trial]["Tail"]["ID1"]

        # merging head and tail data
        merged_datasets = np.hstack(
            (dictionaries[trial]["Head"][recording_type], dictionaries[trial]["Tail"][recording_type]))
        id_length = merged_datasets.shape[1]
        colnames = [f"neuron_{i:03d}" for i in range(id_length)]
        colnames = [dummy if pd.isna(
            ID) else ID for dummy, ID in zip(colnames, id_names)]
        datasets[trial] = pd.DataFrame(merged_datasets, columns=colnames)

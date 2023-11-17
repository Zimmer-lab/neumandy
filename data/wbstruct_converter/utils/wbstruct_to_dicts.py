import scipy.io as scio
import mat73
import os
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import dill
import numpy as np


def contains_substring(text, substring_list):
    """checks if a string contains any of the substrings in a list

    Args:
        text (string): the string to check
        substring_list (list): a list of substrings to check for

    Returns:
        boolean: True if the string contains any of the substrings, False otherwise
    """

    for substring in substring_list:
        if substring.lower() in text.lower():
            return True
    return False


def find_file(root_dir, target_file, include, exclude):
    """walks through a directory and returns a list of all files that match the target file and 
      contain the include string but not the exclude string

    Args:
        root_dir (string): the path to the directory to search
        target_file (string): the file or extension to search for (e.g. wbstruct.mat, .xlsx)
        include (list): list of strings out of which the path should contain at least one
        exclude (list): list of strings that the path must not contain

    Returns:
        list: list of paths to the files that match the criteria
    """
    
    if os.name == 'nt':  # 'nt' indicates Windows
        root_dir = root_dir.replace('\\', '\\\\')
    found_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if any(exclude):
                check = contains_substring(root, include) and not contains_substring(root+f, exclude)
            else:
                check = contains_substring(root, include)
            if f.endswith(target_file) and check:
                found_files.append(os.path.join(root, f))

    return found_files


def load_matlab_file(file_path):
    """loads a matlab file and returns the data

    Args:
        file_path (string): the path to the file   

    Returns:
        dict: the data from the matlab file loaded into a dictionary
    """

    try:
        mat = scio.loadmat(file_path)
    except:
        mat = mat73.loadmat(file_path)
    return mat


def safe_str_contains(text, pattern):
    """safely checks if a string contains a substring

    Args:
        text (string): the string to check
        pattern (string): the substring to check for

    Returns:
        boolean: True if the string contains the substring, False otherwise
    """
    try:
        return pd.Series(text).astype(str).str.contains(pattern, na=False)[0]
    except Exception as e:
        # Handle the error as you see fit (you can print a message or log it)
        print(f"An error occurred: {e}")
        return False


def remove_outer_arrays(arr):
    while isinstance(arr, list):
        arr = arr[0]
    return arr


def get_datasets_dict(root_dir, target_file, include, exclude, recording_type, simple=True):
    """ get a dictionary of dictionaries containing the data from the matlab files and save it to a pickle file

    Args:
        recording_type (string): the type of recording to load from the matlab file (e.g. 'spikes', 'LFP')

    Returns:
        defaultdict: a dictionary of dictionaries containing the data from the matlab files
    """

    datasets = defaultdict(lambda: defaultdict(list))

    print('Searching for paths')
    found_paths = find_file(root_dir, target_file, include, exclude)
    all_paths = [path for path in found_paths if "Quant" in path]
    print('Found {} paths'.format(len(all_paths)))
    
    with_traces = False

    for index in tqdm(range(len(all_paths)), desc="Loading Files"):
        
        # strip the path to the file from the root directory and the target file to get key names
        clean_path = all_paths[index].replace(root_dir+"\\", "")
        clean_path = clean_path.replace("\\Quant\\"+target_file, "")
        clean_path = clean_path.replace(root_dir, "")
        splitted_path = clean_path.split("\\")
        
        # check if the path contains the traces "head" or "tail" for merging later
        with_traces = any(include) and contains_substring("head", include)
        
        if with_traces:
            trace = splitted_path[-1]
            filename = splitted_path[-2]
        else:
            filename = splitted_path[-1]
            
        if "-" in filename:
            filename = filename.replace("-", "")
            filename = filename.replace("_Ctrl", "")
            
        # load the matlab file and get the data
        matfile = load_matlab_file(all_paths[index])
        
        if simple:
            matfile = matfile["simple"]
            
        try:
            recording=remove_outer_arrays(matfile[recording_type])
        except KeyError:
            print("'{}' not found in file".format(recording_type))
            continue
        
        IDs=matfile['ID1']
        
        if with_traces:
            datasets[filename][trace] = {
                recording_type: recording, 'ID1': IDs}
        else:
            datasets[filename] = {
                recording_type: recording, 'ID1': IDs}

    with open('datasets.pkl', 'wb') as file:
        dill.dump(datasets, file)

    return datasets, with_traces




def get_IDs_dict(root_dir, target_file, include, exclude):

    """ get a dictionary containing the IDs from all files

    Returns:
        dict: a dictionary containing the IDs from all files
    """    

    dictofIDs = defaultdict(lambda: defaultdict(list))

    print('Searching for paths')
    all_paths = find_file(root_dir, target_file, include, exclude)
    print('Found {} paths'.format(len(all_paths)))

    for index in tqdm(range(len(all_paths)), desc="Loading Files"):

        excel_sheet = pd.read_excel(
            all_paths[index], sheet_name=None, skiprows=2)

        for key, value in excel_sheet.items():
            if key.startswith('2') and not value.empty:

                colname = 'neuron'
                idcol = 'ID'

                for v in value.columns:
                    if 'ID' in v:
                        idcol = v
                    if 'neuron' in v:
                        colname = v

                # Use safe_str_contains to ignore errors
                pattern = r'\d'
                value['num'] = value[colname].apply(
                    lambda x: safe_str_contains(x, pattern))
                value = value[value['num']]
                value = value.dropna(subset=[colname])
                value = value[idcol].tolist()
                IDs = []
                seen = set()

                for item in value:
                    if item in seen:
                        IDs.append(None)
                    else:
                        seen.add(item)
                        IDs.append(item)

                dictofIDs[key] = IDs

    return dictofIDs

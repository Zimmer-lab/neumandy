import scipy.io as scio
import mat73
import os
import sys
from tqdm import tqdm
from collections import defaultdict
import openpyxl 
import pandas as pd
import dill


def contains_substring(text, substring_list):
    
    for substring in substring_list:
        if substring in text:
            return True
    return False


def find_file_old(start_dir, target_file, include, exclude):
    found_files = []
    for root, dirs, files in os.walk(start_dir):
        if target_file in files and contains_substring(root, include) and not contains_substring(root, exclude):
            found_files.append(os.path.join(root, target_file))
    return found_files


def find_file(start_dir, target_file, include, exclude):
    found_files = []
    for root, dirs, files in os.walk(start_dir):
        for f in files:
            if f.endswith(target_file) and contains_substring(root, include) and not contains_substring(root+f, exclude):
                found_files.append(os.path.join(root, f))
    return found_files


def load_matlab_file(file_path):
    try:
        mat = scio.loadmat(file_path)
    except:
        mat = mat73.loadmat(file_path)
    return mat



def safe_str_contains(text, pattern):
    try:
        return pd.Series(text).astype(str).str.contains(pattern, na=False)[0]
    except Exception as e:
        # Handle the error as you see fit (you can print a message or log it)
        print(f"An error occurred: {e}")
        return False


def get_datasets_dict(root_dir, target_file, include, exclude, recording_type):

    datasets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    print('Searching for paths')
    all_paths = find_file(root_dir, target_file, include, exclude)
    print('Found {} paths'.format(len(all_paths)))

    for index in tqdm(range(len(all_paths)), desc="Loading Files"):
        clean_path = all_paths[index].replace(root_dir+"\\", "")
        clean_path = clean_path.replace("\\Quant\\"+target_file, "")
        splitted_path = clean_path.split("\\")
        trace = splitted_path[-1]
        filename = splitted_path[-2]
        if "-" in filename:
            filename = filename.replace("-", "")
            filename = filename.replace("_Ctrl", "")
        matfile = load_matlab_file(all_paths[index])
        datasets[filename][trace] = {
            recording_type: matfile["simple"][recording_type], 'ID1': matfile["simple"]['ID1']}

    with open('datasets.pkl', 'wb') as file:
        dill.dump(datasets, file)

    return datasets


def get_IDs_dict(root_dir, target_file, include, exclude):

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

                dictofIDs[key] = value[idcol]

    return dictofIDs

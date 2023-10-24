import sys
import wbstruct_to_dicts as wbstruct_dictioniaries
import wbstruct_dicts_to_dataframes as wbstruct_dataframes


directory_path = sys.argv[1]
target_file = sys.argv[2]
include = sys.argv[3]
exclude = sys.argv[4]
recording_type = sys.argv[5]
simple = sys.argv[6]
save_as_hdf5 = sys.argv[7]

if directory_path is None or not directory_path.strip():
    raise ValueError('No Directory Path provided')

datasets = wbstruct_dictioniaries.get_datasets_dict(directory_path,target_file,include,exclude,recording_type,simple) 

dataframes = wbstruct_dataframes.get_dataframes(datasets, recording_type, save_as_hdf5)

print("Available Recordings:",list(dataframes.keys()))
dataframes['Dataset1_20190125_ZIM1428_Ctrl_w2'].head()
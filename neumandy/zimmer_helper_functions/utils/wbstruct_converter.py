import sys
import wbstruct_to_dicts as wbstruct_dictioniaries
import wbstruct_dicts_to_dataframes as wbstruct_dataframes
import dill


directory_path = sys.argv[1]
if sys.argv[2] == '':
    target_file = 'wbstruct.mat'
else:
    target_file = sys.argv[2]
include = sys.argv[3].split(',')
exclude = sys.argv[4].split(',')
if sys.argv[5] == '':
    recording_type = 'deltaFOverF'
else:
    recording_type = sys.argv[5]
if sys.argv[6] == 'y':
    simple = True
else:
    simple = False
if sys.argv[7] == '':
    save_as= 'pkl'
else:
    save_as = sys.argv[7]

if directory_path is None or not directory_path.strip():
    raise ValueError('No Directory Path provided')

datasets, with_2_traces = wbstruct_dictioniaries.get_datasets_dict(directory_path,target_file,include,exclude,recording_type,simple) 

dataframes = wbstruct_dataframes.get_dataframes(datasets, recording_type, with_2_traces, save_as)

print("Available Recordings:",list(dataframes.keys()))


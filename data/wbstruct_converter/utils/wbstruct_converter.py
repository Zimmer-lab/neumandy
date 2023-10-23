import sys

directory_path = sys.argv[1]
target_file = sys.argv[2]
include = sys.argv[3]
exclude = sys.argv[4]
recording_type = sys.argv[5]
simple = sys.argv[6]
save_as_hdf5 = sys.argv[7]

if directory_path is None or not directory_path.strip():
    raise ValueError('No Directory Path provided')

# printing all arguments
# print(f"Argument 1: {directory_path}")
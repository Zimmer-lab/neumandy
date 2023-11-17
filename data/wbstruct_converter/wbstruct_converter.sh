

#!/bin/bash

echo "Converting wbstruct.mat files"


# Prompt the user for input
read -r -p "Directory Path where all recordings are located: " arg1
read -p "Target File: e.g. 'wbstruct.mat'[default] " arg2
read -p "Are there subdirectories that should be included? e.g. 'Head_corr' In case of multiple values, separate them with a comma) " arg3
read -p "Are there subdirectories that should NOT be included? e.g. 'Head' (In case of multiple values, separate them with a comma) " arg4
read -p "Which type of recording should be taken? E.g. deltaFOverF[default], deltaFOverF_bc,.." arg5
read -p "Should the values be taken from the 'simple' struct? y[default]/n " arg6
read -p "Which format should the files be saved as? h5/csv/pkl[default] " arg7

# Call the wbstruct_converter script with the user-provided arguments
python utils/wbstruct_converter.py $arg1 "$arg2" "$arg3" "$arg4" "$arg5" "$arg6" "$arg7"




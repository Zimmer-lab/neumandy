This data folder contains various data that we will run algorithms on.

### Data Sources 

1) Immobilized Whole Brain Imaging Data from Kerem (see project/neurobiology/zimmer/Kerem_Uzel/Whole_brain_imaging/)
2) Immobilized Whole Brain Imaging Data from Rebecca (see project/neurobiology/zimmer/Rebecca) 

1 and 2 used the [whole brain analyzer ](https://github.com/Zimmer-lab/whole_brain_analyzer_MATLAB) to preprocess the raw data from multiple datasets. The resulting mat files are then converted to python readable formats, i. e. hdf5 files and/or a dictionary of pandas dataframes with the **wbstruct_converter** tool. 
**What this data folder contains:**

- wbstruct_converter: a tool with which the data have been obtained from the sources
- data_diagnostics.ipynb: exploration of some transformations of the data (e.g. min-max normalization) 
- exploration_PCA.ipynb: PCA done on raw data
- data_wrangling.ipynb: quantifications to assess necessary preprocessing steps such as data imputation 
- outlier_detection.ipynb: outlier detection using mahanalobis distances and assessing the outliers and their effect on PCA

**What this data folder doesn't contain:**

- the actual data since there is a repo size limit.

### Data Sources 

1) Immobilized Whole Brain Imaging Data from Kerem (see project/neurobiology/zimmer/Kerem_Uzel/Whole_brain_imaging/)
2) Immobilized Whole Brain Imaging Data from Rebecca (see project/neurobiology/zimmer/Rebecca) 

1 and 2 used the [whole brain analyzer ](https://github.com/Zimmer-lab/whole_brain_analyzer_MATLAB) to preprocess the raw data from multiple datasets. The resulting mat files are then converted to python readable formats, i. e. hdf5 files and/or a dictionary of pandas dataframes with the **wbstruct_converter** tool. 

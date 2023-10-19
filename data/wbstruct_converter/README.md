# From MATLAB To Python (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧

Following utility is used to convert MATLAB data to a python readable format. It is specific to the output of the [Whole Brain Analyzer](https://github.com/Zimmer-lab/whole_brain_analyzer_MATLAB) which is a wbstruct.mat file with the variables containing the imaged neurons as continuous time series and the IDs of the neurons (e.g. AVA). 

### **When to use this tool:**

* If the directory that contains all your recordings have following structure:

        +---Recording1
        |   +---Head
        |   |   +---Quant
        |   |   |       wbstruct.mat
        |   |   |       other
        |   +---Tail
        |   |   +---Quant
        |   |   |       wbstruct.mat
        |   |   |       other
        +---Recording2
        |   +---Head
        |   |   +---Quant
        |   |   |       wbstruct.mat
        |   |   |       other
        |   +---Tail
        |   |   +---Quant
        |   |   |       wbstruct.mat
        |   |   |       other

    
    If there are subdirectories in between it should not be a problem<br>
    Example 1: "\my_directory\Dataset1\Head\Quant\wbstruct.mat"<br>
    Example 2: "\my_directory\Cleaned_datasets\Dataset2\Control\Head\Quant\wbstruct.mat"<br>


* If each **wbstruct.mat** file contains the following variables:
    - "simple" (which holds cleaned copies of the raw data)
    - "deltaFOverF" (the normalized fluorescent values)
    - "ID1" (the IDs) 


*Note: Keep in mind that this has not been thoroughly tested and might not work for all cases. If you encounter any problems please open an issue.*

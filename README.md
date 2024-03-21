# ADL
Repo for Wuerth Hackathon. 


## Introduction
This repository contains a comprehensive analysis aimed at identifying potential orsy-shelf customers. We employed three distinct models. 
 1. Anonmaly detection
 2. Cosine Similarity
 3. Gradient Boosting

In the end we analysed the results and indicated how many of the models (1, 2 or 3) identified a potential orsy-shelf customer. The analysis was done for four different datasets (see presentation slides).


## Installation
It is recommended to use python 3.11.6 to conduct the analysis. All packages necessary are listed in the requirements.txt file and can be installed via `pip install -r requirements.txt`.


## Structure
### 00_docs
This folder contains the xslx file which lists all variables provided and a short explanation.

### 01_data
This folder contains three elements:
 1. **data_model_eval**: A folder containing the optimal hyperparameters for the gradient boosting for each of the four datasets, computed in `02_code/03_model_selection.ipynb`.
 2. **data_out**: The anomaly counts for varying thresholds for each of the four datasets, as well as the csv files containing information which customer has been predicted to be interested in an orsy-shelf. All are computed in `02_code/04_orsy_identification`
 3. **dataset_wurth.csv**: A csv file cotaining the provided data.

 ### 02_code
 This folder contains all scripts needed to perform the analysis:
 1. **adl**: A package containing all classes and functions built.
    * compare_results: A script containing the function `compare_results()` to compare the identified potential customers within a dataframe across the different models used
    * data_processor: A script containing the class `DataProcessor` to prepare and process the data for our analysis.
    * models: A script containing the classes `AnomlayDetector`, `CosineSimilarityCalculator` and `PropensityScorer` for the models used in `02_code/04_orsy_identification.ipynb`
    * ps_model_eval: A script containing a class to determine the best model for the propensity scoring model (which is one of the three models used to identify potential customers) and another class which tunes the hyperparameters of the best model (Gradient Boosting).
2. **01_initial_data_prep**: A jupyter notebook in which multicollinearity is detected to exclude problematic variables in the `DataProcessor` class.
3. **02_descriptive_data**: A jupyter notebook containing descriptive statistics as well as some analysis used to justify constructing datasets in the `DataProcessor` class that only contain a fraction of the features offered in the original dataset.
4. **03_model_selection**: A jupyer notebook containing the steps to identify a suitable propensity scoring model for each of the four datasets. 
5. **04_orsy_identidication**: A jupyer notebook conducting the actual identification of potential orsy-shelf customers
6. **05_post_descriptive_data**: A jupyter notebook conducting a concluding analysis to identify common traits, etc. of potential orsy-shelf customers.

### 03_report
This folder contains two elements:
 1. **Praesentation**: A pdf file containing the slides for the presentation.
 2. **graphs**: A folder containing the plots used in the presentatoion.


## requirements
The requirements file listing all packages needed.
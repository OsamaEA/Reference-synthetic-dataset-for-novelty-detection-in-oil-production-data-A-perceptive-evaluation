#!/usr/bin/env python
# coding: utf-8

# Main librarires
import numpy as np
np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import time
from functools import reduce
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
# Scaling necessity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# Top metods mentioned in previous papers
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.abod import ABOD

# New Methods investigated by us and worked fine
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.sos import SOS

from pyod.utils.utility import standardizer
from pyod.models.combination import aom, moa, average, maximization, majority_vote, median
from pyod.utils.utility import argmaxn
from pyod.utils.utility import score_to_label

# removing warnings
import warnings
warnings.filterwarnings("ignore")


# # Data Loaders
# Wells original names as obtained from UNISIM wells
# Source of data: https://www.unisim.cepetro.unicamp.br/benchmarks/br/
UNISIM_I_wells = ['NA1A', 'NA2', 'NA3D', 'RJS19', 'PROD005', 'PROD008', 'PROD009', 'PROD010', 
                  'PROD012', 'PROD014', 'PROD021', 'PROD023A', 'PROD024A', 'PROD025A']
UNISMI_II_wells = ['wildcat', 'PROD1', 'PROD2', 'PROD3', 'PROD4', 'PROD5', 
                   'PROD6', 'PROD7', 'PROD8', 'PROD9', 'PROD10']
UNISIM_III_wells = ['P11', 'P12', 'P13', 'P14', 'P15', 'P16|']

df = pd.read_excel('/root/capsule/data/original_UNISIM_dataset.xlsx', header = [0,1], index_col = [0], sheet_name = 'synthetic_data')
print('Started the code and Loaded data successfully!')

# # 1. Data Wrangling
def prepare_dataset(df):
    '''
        prepare dataset for cleaning in the required format of oil and days names
        INPUT:
             the input dataset should have the following columns {Oil Rate SC, Water Rate SC, Liquid Rate SC,
                                                                  Gas Rate SC, Well Pressure}
        OUTPUT:
            the output dataset is clean dataset with {Qo_bpd, WC_porc, GOR_cfd_per_bpd, Rate_normalized_pressure}
                                                                                    '''
    # creating an empty list for dataframes appendence
    df_ready = []
    for i, well_name in enumerate(df.columns.get_level_values(0).unique().tolist()):
        df_trial = df[well_name]
        # Renamin columns and index names
        df_trial = df_trial.rename(columns={'Oil Rate SC': 'Qo_bpd'})
        df_trial.index.names = ['days']
        # Making some calculated parameters
        df_trial.loc[:, 'WC_perc'] = df_trial.loc[:, 'Water Rate SC'] *100 / df_trial.loc[:, 'Liquid Rate SC']
        df_trial.loc[:, 'GOR_cfd_per_bpd'] = df_trial.loc[:, 'Gas Rate SC'] / df_trial.loc[:, 'Qo_bpd']
        df_trial.loc[:, 'Rate_normalized_pressure'] = df_trial.loc[:, 'Well Pressure'] / df_trial.loc[:, 'Qo_bpd']
        # Dropping the original columns that i no longer need
        df_trial.drop(['Water Rate SC', 'Gas Rate SC', 'Liquid Rate SC', 'Well Pressure'], axis = 1, inplace = True)
        df_ready.append(df_trial)
        # Setting multi index columns
        df_trial.columns = pd.MultiIndex.from_product([[well_name], df_trial.columns])
    # We do not drop any values so we just concat dataframes since there is no missing data
    df_final = pd.concat(df_ready, axis = 1)
    return(df_final)


def add_white_noise(df, perc_outliers):
    '''
        INPUT: 
              dataset of normal production without any anomalies
        OUTPUT:
             dataset with added noise to the oil production according to the percentage specified
                                                                                                 '''
    # Making a copy of the dataset to work on while prevailing the original dataframe for future use
    df_outliers = df.copy()
    # Adding column of ground truth assuming it all as 0 or not outliers
    df_outliers.loc[:, 'ground_truth'] = 0        
    # Making a copy of production column
    df_outliers['Qo_bpd_original'] = df_outliers['Qo_bpd']
    # Replacing nan values with zero oil productions
    df_outliers['Qo_bpd_original'] = df_outliers['Qo_bpd_original'].replace(np.nan, 0)
    # Using the concept of rolling average / minimum / maximum for white noise addition criteria
    df_outliers['rolling_qo'] = df_outliers['Qo_bpd_original'].rolling(3, center = True, axis=0).mean()
    df_outliers['rolling_qo_max'] = df_outliers['Qo_bpd_original'].rolling(3, center = True, axis=0).max()
    df_outliers['rolling_qo_min'] = df_outliers['Qo_bpd_original'].rolling(3, center = True, axis=0).min()
    # replacing null values of the rolling production with the corresponding actual values
    # those null values would evolve at the edge of windows
    df_outliers.loc[:, 'rolling_qo'] = np.where(df_outliers.loc[:, 'rolling_qo'].isnull() == True,
                                                    df_outliers.loc[:, 'Qo_bpd_original'], df_outliers.loc[:, 'rolling_qo']) 
    df_outliers.loc[:, 'rolling_qo_max'] = np.where(df_outliers.loc[:, 'rolling_qo_max'].isnull() == True,
                                                    df_outliers.loc[:, 'Qo_bpd_original'], df_outliers.loc[:, 'rolling_qo_max'])
    df_outliers.loc[:, 'rolling_qo_min'] = np.where(df_outliers.loc[:, 'rolling_qo_min'].isnull() == True,
                                                    df_outliers.loc[:, 'Qo_bpd_original'], df_outliers.loc[:, 'rolling_qo_min'])
    # Setting noise levels
    # Levels for production rates less than 300
    thr_300 = np.random.normal(0.30, 0.03, len(df_outliers))
    # Levels for production rates between 300 and 1000
    thr_1000 = np.random.normal(0.25, 0.03, len(df_outliers))
    # Levels for production rates between 1000 and 3000
    thr_3000 = np.random.normal(0.20, 0.03, len(df_outliers))
    # Levels for production rates between 3000 and 6000
    thr_6000 = np.random.normal(0.15, 0.03, len(df_outliers))
    # Levels for production rates higher than 6000
    thr = np.random.normal(0.10, 0.03, len(df_outliers))
    
    # Setting possibility that outliers could either be positive or negative
    df_outliers.loc[:, 'random'] = np.random.choice([-1,1], len(df_outliers)) 
    # Generating white gaussian noise around SD value from each value of the coil production
    df_outliers.loc[:, 'std'] = np.select([df.Qo_bpd < 300,
                                           ((df.Qo_bpd >= 300) & (df.Qo_bpd < 1000)),
                                           ((df.Qo_bpd >=1000) & (df.Qo_bpd < 3000)),
                                           ((df.Qo_bpd >=3000) & (df.Qo_bpd < 6000))],
                                          [thr_300, thr_1000, thr_3000, thr_6000],
                                          default = thr) # for default production rates higher than 6000
    # Setting new production rates for all corresponding values according to the given criteria
    df_outliers['Qo_bpd'] = df_outliers['rolling_qo'] * (1 + (df_outliers['std']*df_outliers.loc[:, 'random']))
    # Setting production rate data type to integers
    df_outliers['Qo_bpd'] = df_outliers['Qo_bpd'].astype(int)
    # In case the normal distribution resulted in a negative value, we replace it by 25 to avoid it being removed
    # 25 is arbitarily selected as a minimum value for production rate that is not zero.
    df_outliers['Qo_bpd'][df_outliers['Qo_bpd'] <=0] = 25
    
    # Rolling values minimum and maximum thresholds
    lower_300, upper_300 = 0.25, 0.35
    lower_1000, upper_1000 = 0.20, 0.30
    lower_3000, upper_3000 = 0.15, 0.25
    lower_6000, upper_6000 = 0.10, 0.20
    lower, upper = 0.05, 0.15
    
    # Creating two additional columns to set lower and higher boundaries values corresponding to the created oil rates
    # Lower boundary
    df_outliers.loc[:, 'lower_boundary'] = np.select([df.Qo_bpd < 300,
                                                      ((df.Qo_bpd >= 300) & (df.Qo_bpd < 1000)),
                                                      ((df.Qo_bpd >=1000) & (df.Qo_bpd < 3000)),
                                                      ((df.Qo_bpd >=3000) & (df.Qo_bpd < 6000))],
                                                     [lower_300, lower_1000, lower_3000, lower_6000],
                                                     default = lower)
    # Upper boundary
    df_outliers.loc[:, 'upper_boundary'] = np.select([df.Qo_bpd < 300, ((df.Qo_bpd >= 300) & (df.Qo_bpd < 1000)),
                                           ((df.Qo_bpd >=1000) & (df.Qo_bpd < 3000)),
                                           ((df.Qo_bpd >=3000) & (df.Qo_bpd < 6000))],
                                          [upper_300, upper_1000, upper_3000, upper_6000], default = upper)    
    
    # To consider any generated value in range of +/- tolerance% as inlier not outlier as a lower boundary
    # Boundaries are selected based on a window such that neighbouring points are considered when a point is assumed outlying
    df_outliers.loc[:, 'ground_truth'] = np.where(df_outliers.loc[:, 'Qo_bpd'] >
                                        (1 + df_outliers.loc[:, 'lower_boundary']) * df_outliers.loc[:, 'rolling_qo_max'],
                                         1,
                                        (np.where(df_outliers.loc[:, 'Qo_bpd'] > 
                                        (1 - df_outliers.loc[:, 'lower_boundary']) * df_outliers.loc[:, 'rolling_qo_min'],
                                         0,
                                         1)))
    # Set a new outliers dataset to be the actual generated anomalies only
    df_outliers = df_outliers[df_outliers['ground_truth'] == 1]   
    # To avoid any points that are so far from the original values and that could be easy to catch, we set upper boundaries
    # Similar to the above methodology
    df_outliers.loc[:, 'ground_truth'] = np.where(df_outliers.loc[:, 'Qo_bpd'] >
                                        (1 + df_outliers.loc[:, 'upper_boundary']) * df_outliers.loc[:, 'rolling_qo_max'],
                                         0,
                                        (np.where(df_outliers.loc[:, 'Qo_bpd'] >
                                        (1 - df_outliers.loc[:, 'upper_boundary']) * df_outliers.loc[:, 'rolling_qo_min'],
                                         1,
                                         0)))  
    
    # The above criteria could be briefly summarized as:
    # A point is outlier if:
    # 1) (1+Upper boundary)* max window Qo  > value > (1+lower boundary)* max window Qo 
    # 2) (1-Upper boundary)* min window Qo  < value < (1-lower boundary)* min window Qo
    
    
    # Sorting index of outliers and remove those which are tolerated as easy targets
    df_outliers = df_outliers[df_outliers['ground_truth'] == 1]
    
    # Selection a percentage from outlying points according to the pre-defined value
    perc_to_remove = perc_outliers * len(df) / len(df_outliers)
    # if the requested percentage of oultiers is higher than the generated outlying dataset, use it as it is
    if perc_to_remove > 1:
        perc_to_remove = 1
    # sampling the outlying dataset as per requested percentage
    df_outliers = df_outliers.sample(frac = perc_to_remove)
    # Correct number of outliers after considering boundaries
    new_perc = len(df_outliers) / len(df)
    return(df_outliers, new_perc)


def clean_data_same(df, perc_outliers, seed_num):
    '''
        Change some data into outliers without adding new dataset as outliers
        INPUT:
              df: the original clean dataset with the following columns [Qo_bpd, online_days"if True"]
              outliers_sd_value: value of STD for white gaussian noise from each point
              perc_outliers: fraction of outliers in the dataset
              tolerance: the tolerance +/- below which an outlier would still be conisdered an inlier
                                                                                        '''
    # To get same output in random space everytime based on a given input
    np.random.seed(seed_num)
    df_ready = []
    outliers_indices = []
    new_outlier_clean_values = []
    for i, well_name in enumerate(df.columns.get_level_values(0).unique().tolist()):
        df_trial = df[well_name]
        # Remove nan or zero production values from the dataset
        df_trial = df_trial[df_trial['Qo_bpd'].notna()]
        df_trial = df_trial[df_trial['Qo_bpd'] != 0]
        
        # Setting clean and outlier dataframes with their groudn truths
        df_clean = df_trial.copy()
        # setting ground truth to zero as they are originally inliers
        df_clean.loc[:, 'ground_truth'] = 0
        
        # Adding white noise and making outliers dataframe
        # Making the random generator completely random to avoid any bias  
        np.random.seed(int(np.random.normal(1000, 500 * perc_outliers)))
        # Using a pre-defined function to add white noise
        df_outliers, new_perc = add_white_noise(df_trial, perc_outliers)
        # Getting the index of outliers days sorted
        outliers_index = sorted(df_outliers.index.tolist())
        # clean dataframes
        df_clean = df_clean[~df_clean.index.isin(outliers_index)]
        # Merged dataframe
        df_merge = pd.concat([df_clean, df_outliers], axis = 0)
        # Sorting index for the total dataframe now including inliers and outliers
        df_trial = df_merge.sort_index()
        
        # To avoid any error of negative values in the dataset
        df_trial['Qo_bpd'][df_trial['Qo_bpd'] < 0] = 0
        # Replacing errors with nans as they would be data with no production available
        df_trial = df_trial.replace([np.inf, -np.inf, 0], np.nan)
        # Setting index values
        df_trial.index.names = ['days']
        # Replacing nan values of ground truth as zeros as we will measure against it (we will not consider them in evaluation)
        df_trial['ground_truth'] = df_trial['ground_truth'].replace([np.nan], 0)   
        
        # Calculating some variables that we might need to use later
        df_trial.loc[:, 'Qo_cum_bbl'] = df_trial.loc[:, 'Qo_bpd'].cumsum()
        
        # Setting multi index columns
        df_trial.columns = pd.MultiIndex.from_product([[well_name], df_trial.columns])
        
        # Appending dataframes and indices
        df_ready.append(df_trial)
        outliers_indices.append(outliers_index)
        new_outlier_clean_values.append(new_perc)
    # Merging dataframes on idnex column so that we dont drop nan values by mistake
    df_new = reduce(lambda df_1,df_2 : pd.merge(df_1,df_2, how = 'outer', left_index = True, right_index = True), df_ready)
    
    return(df_new, outliers_indices, new_outlier_clean_values)


# # 2. Processing

def outliers_plotting(df_plot_full_cols, df_trial, model, perc, ground_truth, original_days, original_outliers_indices_days, technique_name, evaluation_matrix, x_col_plot, y_col_plot):
    '''
        INPUT:
              df_plot_full_cols: original dataset with NaNs and inf data
              df_trial: clean dataset that is sued in the model of outliers detection
              preds: predictions that resulted from OD modelling
              original_days: the list of days index list
              original_outliers_indices_days: original outliers that we added
              techniqu_name: the name of the model you used
              evaluation_matrix: list that would append scores
        OUTPUT:
              a plot that only shows clean data after plotting and distinguishes false negative points on the plot that
              are detected as clean while in real they are outliers
                                                                                                                      '''
    t_start = time.time()
    # Checking if it is a combination mode or not
    if type(model) == list:
        # number of classifiers
        n_clf = len(model)
        
        # Assigning name for each model
        model_1 = model[0]
        model_2 = model[1]
        model_3 = model[2]
        model_4 = model[3]
        
        # fitting models and raw scores that will append the scores in
        raw_scores = np.zeros([df_trial.shape[0], n_clf])
        raw_preds = np.zeros([df_trial.shape[0], n_clf])
        model_1_fit = model_1.fit(df_trial)
        model_2_fit = model_2.fit(df_trial)
        model_3_fit = model_3.fit(df_trial)
        model_4_fit = model_4.fit(df_trial)
        
        # COMBO averaging
        model_1_scores = model_1_fit.decision_scores_
        model_2_scores = model_2_fit.decision_scores_        
        model_3_scores = model_3_fit.decision_scores_
        model_4_scores = model_4_fit.decision_scores_    
     
        # Setting raw scores of each algorithm
        raw_scores[:, 0] = model_1_scores
        raw_scores[:, 1] = model_2_scores        
        raw_scores[:, 2] = model_3_scores
        raw_scores[:, 3] = model_4_scores

        # Standardizing results
        raw_scores_norm = standardizer(raw_scores)
        
        # use maximization matrix with scores
        scores = maximization(raw_scores_norm)
        # prediction labels coversion
        preds = score_to_label(scores,perc)
    
        
    else:
        # For single algorithms 
        # Fit the model
        model_fit = model.fit(df_trial)
        # Predict the labels at the current contamination fraction
        preds = model_fit.labels_
        # Calculate scores
        scores = model_fit.decision_scores_
        
      
    ################################### SCORING PURPOSE ##########################################
    # For plotting roc curve if needed
    fpr, tpr, thres = roc_curve(ground_truth, scores)
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(ground_truth, preds).ravel()
    # Evaluation matrix over AUC and maP
    auc_score = roc_auc_score(ground_truth, scores)
    average_precision = average_precision_score(ground_truth, scores)
    # Cohen kappa and matthew coefficient
    cohen_kappa = cohen_kappa_score(ground_truth, preds)
    phi_coef = matthews_corrcoef(ground_truth, preds)
    # Traditional evaluation matrix
    Precision, Recall, f_beta_score, _ = precision_recall_fscore_support(ground_truth, preds, average = 'binary', beta = 1)
    Accuracy = accuracy_score(ground_truth, preds)
    Fallout = fp / (fp + tn)
    
    ################################### PLOTTING PURPOSE ##########################################
    # Making a copy of the dataframe to avoid overriding new columns
    df_new = df_trial.copy()
    # Adding original days to avoid it being considered into the algorithm
    df_new['original_days']= original_days
    # clean indices, clean dataframe scaled, and clean days
    abn_ind = np.where(pd.DataFrame(preds) == 0)[0].tolist()
    df_plot_clean_unique = df_new[df_new.index.isin(abn_ind)]
    unique_clean_days = df_plot_clean_unique['original_days']
    # outlier indices, outlier dataframe scaled, and outlier days
    out_ind = np.where(pd.DataFrame(preds) == 1)[0].tolist()
    df_plot_outliers_unique = df_new[df_plot_full_cols.index.isin(out_ind)]
    unique_outlier_days = df_plot_outliers_unique['original_days']
    
    # clean and outlier dataframes according to algorithm 
    df_plot_full_clean = df_plot_full_cols[df_plot_full_cols['days'].isin(unique_clean_days)]
    df_plot_full_outlier = df_plot_full_cols[df_plot_full_cols['days'].isin(unique_outlier_days)]
    # Original clean and outlier dataframe
    df_plot_full_clean_original = df_plot_full_cols[~df_plot_full_cols['days'].isin(original_outliers_indices_days)]
    df_plot_full_outlier_original = df_plot_full_cols[df_plot_full_cols['days'].isin(original_outliers_indices_days)]
    # different dataframes used later
    df_plot_full_true_outlier = pd.merge(df_plot_full_outlier, df_plot_full_outlier_original['days'], how='inner', on=['days'])
    df_plot_full_false_outlier = pd.merge(df_plot_full_outlier, df_plot_full_clean_original['days'], how='inner', on=['days'])        
    df_plot_full_true_clean = pd.merge(df_plot_full_clean, df_plot_full_clean_original['days'], how='inner', on=['days'])
    df_plot_full_false_clean = pd.merge(df_plot_full_clean, df_plot_full_outlier_original['days'], how='inner', on=['days'])
    
    # Uncomment the below block if code if plotting was required simultatneously
    # Plotting     
    #plt.scatter(df_plot_full_clean[x_col_plot], df_plot_full_clean[y_col_plot], marker = "X", color = 'green')
    #plt.scatter(df_plot_full_false_clean[x_col_plot], df_plot_full_false_clean[y_col_plot], marker = "X", color = 'red')
    # Title and labels
    #plt.title(str(technique_name) + '-' + str(round(f_beta_score,2)))
    #plt.xlabel(x_col_plot)
    #plt.ylabel(y_col_plot)
    
    # Execution time calculation
    t_finish = time.time()
    execution_time = round((t_finish - t_start),2)
    
    # Standard ealuation matrix
    matrix = [f_beta_score, Accuracy, Recall, Precision, Fallout,
              cohen_kappa, phi_coef, auc_score, average_precision, execution_time]
    # Appending all scores to evaluation matrix
    evaluation_matrix.append(matrix)
        
    return(evaluation_matrix)

def final_table(outlier_perc, COMBO_matrix, ABOD_matrix, SOS_matrix, KNN_matrix_avg, LOF_matrix, IFor_matrix,
                          COF_matrix, KNN_median_matrix):
    '''
        INPUT: evaluation matrices as lists
        OUTPUT: table of evaluation matrices
                                                                                                        '''
    # Setting a dataframe that contains all evaluation metrics
    df = pd.DataFrame([np.mean(COMBO_matrix, axis = 0).tolist(), np.mean(ABOD_matrix, axis = 0).tolist(),
                        np.mean(SOS_matrix, axis = 0).tolist(), np.mean(KNN_matrix_avg, axis = 0).tolist(),
                        np.mean(LOF_matrix, axis = 0).tolist(), np.mean(IFor_matrix, axis = 0).tolist(),
                        np.mean(COF_matrix, axis = 0).tolist(), np.mean(KNN_median_matrix, axis = 0).tolist()],
                        columns = ['f2_score', 'Accuracy', 'Recall', 'Precision', 'Fallout', 'cohen_kappa',
                                   'matthews_phi_coef', 'auc_score', 'maP', 'execution_time'],
                        index = ['COMBO', 'ABOD', 'SOS', 'KNN', 'LOF', 'iForest', 'COF', 'KNN_median'])
    # Setting Multi index header with Standard deviation and outliers fractions
    df.columns = pd.MultiIndex.from_product([[('{} Outliers').format(outlier_perc)], df.columns])
    
    return(df)

def compare_OD_methods(df_original, percentage, seed_number, columns_to_include = 'all', x_col_plot = 'days', y_col_plot = 'Qo_bpd', scaling = True):
    '''
        INPUT:
             df_original: dataframe or table that might include one or multiple wells data
             columns_to_include: features that might be related to outliers recognition, it should be a list.. default is all
             x_col_plot: the first axis you want to visualize horizontally.. default is time
             y_col_plot: the second axis you want to visualize vertically.. deafult is Qo
             
        OUTPUT:
             a graph that includes the original data as well as how it looks like after cleaning
             
        NOTE:
            The input table should have multiindex columns level and the higher level is of wells' names
                                                                                                     '''
    # Reading original outliers or groudn truth and the dataframe cleaned
    df, outliers_original, percentages_outliers_clean = clean_data_same(df_original, percentage, seed_number)

    COF_matrix = []
    LOF_matrix = []
    IFor_matrix = []
    ABOD_matrix = []
    KNN_matrix_avg = []
    SOS_matrix = []
    KNN_median_matrix = []
    COMBO_matrix = []
    
    df_reference_seed_42 = []
    for i, well_name in enumerate(df.columns.get_level_values(0).unique().tolist()):
        #print('*****************************************   {}   ********************************************'.format(well_name))
        # Setting original dataframe with no outliers
        origin_no_out_df = df_original[well_name]
        origin_no_out_df = origin_no_out_df[origin_no_out_df['Qo_bpd'].notna()]
        origin_no_out_df = origin_no_out_df.reset_index()
        
        # The modfified percentage outlying value calculated from clea_data_same function
        perc = percentages_outliers_clean[i]
        # Outliers indices that we have created
        original_outliers_indices_days = outliers_original[i]
        # New dataframe for each well
        df_well = df[well_name]
        # keeping points with available oil production only
        df_well = df_well[df_well['Qo_bpd'].notna()]
        # Ground truth
        ground_truth = df_well['ground_truth']
        # replace inf points into NaNs
        df_well = df_well.replace([np.inf, -np.inf], np.nan)
        
        #ORIGINAL PLOT SECTION
        # the full dataset to be used in plotting that includes features not used in modelling
        df_plot_full_cols = df_well.copy()
        # We keep null values as nans in the plotting
        df_plot_full_cols = df_plot_full_cols.reset_index()
        # Outlier df for plotting
        outlier_df = df_plot_full_cols[df_plot_full_cols['days'].isin(original_outliers_indices_days)]
        origin_no_out_outliers_df = origin_no_out_df[origin_no_out_df['days'].isin(original_outliers_indices_days)]
        
        # replace NaNs with zeros to use in modelling
        df_well = df_well.reset_index().fillna(0) 
        
        original_days = df_well['days']
        if scaling == True:
            # Scaling dataset with MinMax technique
            df_well = pd.DataFrame(MinMaxScaler().fit_transform(df_well), columns = df_well.columns)
        
        # Selecting features for each well that will build the model for outliers detection
        if columns_to_include == 'all':
            # using the entire dataset attributes for novelty detection
            df_trial = df_well
        else:
            # using the pre-define attributes for novelty detection
            cols = columns_to_include.copy()
            if 'days' not in cols:
                # Appending days in case that the selected features does not include it
                cols.append('days')
            # setting dataset with specified attributes
            df_trial = pd.DataFrame(df_well[cols])
       
        ''' ORIGINAL SCATTER PLOT'''
        # Full table for original scatter plotting
        df_full = df.copy()
        df_full = df_full[well_name].reset_index()
        # Making the original plot
        #fig = plt.figure(figsize = (25,10))
        
        # Uncomment the below block of code for data visualization
        ###################################################################################################
        # Plotting the original data with points to be modifed as outliers identified
        #ax = fig.add_subplot(2,5,1)
        #plt.scatter(x = origin_no_out_df[x_col_plot], y= origin_no_out_df['Qo_bpd'])
        #plt.scatter(x = origin_no_out_outliers_df[ x_col_plot], y= origin_no_out_outliers_df[y_col_plot], label = 'outliers', color = 'orange')
        #plt.title('Original Plot')
        #plt.xlabel(x_col_plot)
        #plt.ylabel(y_col_plot)
        ###################################################################################################
        
        # Uncomment the below block of code for data visualization
        ###################################################################################################
        # Plotting inliers and outliers on the graph
        #ax1 = fig.add_subplot(2,5,2, sharex = ax, sharey = ax)
        #plt.scatter(x = df_full[x_col_plot], y= df_full[y_col_plot], label = 'inliers')
        #plt.scatter(x = outlier_df[ x_col_plot], y= outlier_df[y_col_plot], label = 'outliers', color = 'red')
        #plt.title('Original Plot with outliers')
        #plt.xlabel(x_col_plot)
        #plt.ylabel(y_col_plot)  
        ###################################################################################################
        
        '''initiatory'''
        # uncomment the below line for visualization simulatneously
        #ax3 = fig.add_subplot(2, 5, 3, sharex = ax, sharey = ax)
        # 4 algorithms used for building the ensemble model
        combo_clf1 = ABOD(contamination = perc, n_neighbors = 5)
        combo_clf2 = ABOD(contamination = perc, n_neighbors = 6)
        combo_clf3 = KNN(contamination = perc, n_neighbors = 4, method = 'median')
        combo_clf4 = SOS(contamination = perc, perplexity = 12)
        # using the outliers_plotting function
        COMBO_matrix = outliers_plotting(df_plot_full_cols, df_trial, 
                                         [combo_clf1, combo_clf2, combo_clf3, combo_clf4], perc, 
                                         ground_truth, original_days, original_outliers_indices_days, 'COMB', COMBO_matrix, x_col_plot, y_col_plot)
  
        '''URTEC-208384-MS'''
        # uncomment the below line for visualization simulatneously
        #ax4 = fig.add_subplot(2, 5, 4,  sharex = ax, sharey = ax)
        Abod_clf = ABOD(contamination = perc, n_neighbors = 5)
        ABOD_matrix = outliers_plotting(df_plot_full_cols, df_trial, Abod_clf, perc, ground_truth, original_days, original_outliers_indices_days, 'ABOD', ABOD_matrix, x_col_plot, y_col_plot)

        '''initiatory'''
        # uncomment the below line for visualization simulatneously
        #ax5 = fig.add_subplot(2, 5, 5, sharex = ax, sharey = ax)
        SOS_clf = SOS(contamination = perc, perplexity = 12)
        SOS_matrix = outliers_plotting(df_plot_full_cols, df_trial, SOS_clf, perc, ground_truth, original_days, original_outliers_indices_days, 'SOS', SOS_matrix, x_col_plot, y_col_plot)

        '''https://doi.org/10.1016/j.eswa.2021.116371'''
        # uncomment the below line for visualization simulatneously
        #ax6 = fig.add_subplot(2, 5, 6, sharex = ax, sharey = ax)
        KNN_clf = KNN(contamination = perc, n_neighbors = 10, method = 'mean')
        KNN_matrix_avg = outliers_plotting(df_plot_full_cols, df_trial, KNN_clf, perc, ground_truth, original_days, original_outliers_indices_days, 'KNN avg', KNN_matrix_avg, x_col_plot, y_col_plot)


        ''' SPE-179958-MS '''
        # uncomment the below line for visualization simulatneously
        #ax7 = fig.add_subplot(2, 5, 7, sharex = ax, sharey = ax)
        LOF_clf = LOF(contamination = perc, n_neighbors = 30, metric = 'euclidean', p = 1, novelty = False)
        LOF_matrix = outliers_plotting(df_plot_full_cols, df_trial, LOF_clf, perc, ground_truth, original_days, original_outliers_indices_days, 'LOF', LOF_matrix, x_col_plot, y_col_plot)
        
        '''A Visual Analytics Approach to Anomaly Detection in Hydrocarbon Reservoir Time Series Data'''
        # uncomment the below line for visualization simulatneously
        #ax8 = fig.add_subplot(2, 5, 8, sharex = ax, sharey = ax)
        IForest_clf = IForest(contamination = perc, n_estimators = 100, behaviour = 'new', random_state = 42)
        IFor_matrix = outliers_plotting(df_plot_full_cols, df_trial, IForest_clf, perc, ground_truth, original_days, original_outliers_indices_days, 'iForest', IFor_matrix, x_col_plot, y_col_plot)

        # uncomment the below line for visualization simulatneously
        #ax9 = fig.add_subplot(2, 5, 9, sharex = ax, sharey = ax)
        COF_clf = COF(contamination = perc, n_neighbors = 7)
        COF_matrix = outliers_plotting(df_plot_full_cols, df_trial, COF_clf, perc, ground_truth, original_days, original_outliers_indices_days, 'COF', COF_matrix, x_col_plot, y_col_plot)
        
        # uncomment the below line for visualization simulatneously
        #ax10 = fig.add_subplot(2, 5, 10, sharex = ax, sharey = ax)
        KNN_median_clf = KNN(contamination = perc, n_neighbors=4, method = 'median')
        KNN_median_matrix = outliers_plotting(df_plot_full_cols, df_trial, KNN_median_clf, perc, ground_truth, original_days, original_outliers_indices_days, 'KNN', KNN_median_matrix, x_col_plot, y_col_plot)
         
        # uncomment the below line for simultaneous plotting
        #plt.show()
        #print ('-'*120)  
        
        # Appending reference dataset at seed 42
        indx_df_reference_seed_42 = df_trial[['days', 'Qo_bpd']]
        # Removing index name
        indx_df_reference_seed_42 = indx_df_reference_seed_42.rename_axis(None, axis=1)
        # Adding a multiindex for wellname with attribute name
        indx_df_reference_seed_42.columns = pd.MultiIndex.from_product([[well_name],
                                                                        indx_df_reference_seed_42.columns])
        # Adding a top-pevel of outlying percentage
        indx_df_reference_seed_42=pd.concat([indx_df_reference_seed_42],keys=['Outliers - {}'.format(percentage)],axis=1)
        # Adding a top-pevel of original dataset name
        indx_df_reference_seed_42=pd.concat([indx_df_reference_seed_42],keys=['Modified {}'.format(
        ['UNISIM I' if well_name in UNISIM_I_wells else 'UNISIM II' if well_name in UNISMI_II_wells else 'UNISIM III'][0])],axis=1)

        # Appending the new dataframe to the empty list
        df_reference_seed_42.append(indx_df_reference_seed_42)
        
        
    # Extracting final table of conusion matrix results
    df_final = final_table(percentage, COMBO_matrix, ABOD_matrix, SOS_matrix, KNN_matrix_avg, LOF_matrix, IFor_matrix,
                          COF_matrix, KNN_median_matrix)
    # Merging dataframes
    df_reference_seed_42_full = reduce(lambda df_1,df_2 : pd.merge(df_1,df_2, how = 'outer', left_index = True, right_index = True), df_reference_seed_42)    
    return(df_reference_seed_42_full, df_final)


# # 3. Results Extraction
df_prep = prepare_dataset(df)


# #### A) Reference Dataset and Benchmark Results at Seed Number of 42
# Creating empty lists
reference_dataset_ori = []
output_results_ori = []
# Looping through different outlying percentages
for percnt in np.arange(0.05, 0.50, 0.05):
    percnt = round(percnt, 2)
    # Extracting outlying dataset and results metrics on it
    ref_df, out_df = compare_OD_methods(df_prep, percnt, 42, ['days', 'Qo_bpd'], scaling = False)
    # Appending datasets
    reference_dataset_ori.append(ref_df)
    output_results_ori.append(out_df)
print('Successfully created Reference-Synthetic Dataset and benchmarked algorithms performance at Random Seed 42!')
# Concating datasets
reference_dataset = pd.concat(reference_dataset_ori, axis = 1)
output_results = pd.concat(output_results_ori, axis = 1)
# Convertign to excel sheets for future use as benchmark
reference_dataset.to_excel('/root/capsule/results/Reference_synthetic_dataset_for_novelty_detections.xlsx')
output_results.to_excel('/root/capsule/results/Results_on_reference_synthetic_dataset_seed_42.xlsx')

# #### B) Benchmark Results on 100 randomly generated dataset
# Results at 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, and 45% constant outlying percentage
outlying_pcts = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
df_total_bootstrap = []
print('Started evaluating algorithms performance across 100 different random states for higher confidence with interpretation!')
# Looping results through 100 different seed number
for outlying_pct in outlying_pcts:
    # Printing loop update every 50 seed number
    print('Looping outlying percentage of {}!'.format(outlying_pct))
    # initializing the empty list
    df_all_pct = []
    for seed_num in range(0,202,2):
        _, df_otly = compare_OD_methods(df_prep, outlying_pct, seed_num, ['days', 'Qo_bpd'])
        df_all_pct.append(df_otly)
    # Appedning dataset of all random generators for each single percentage
    df_pct = pd.concat(df_all_pct, axis = 0)
    # Appending the output dataset into one list
    df_total_bootstrap.append(df_pct)
print('Successfully evaluated algorithms performance across 100 different random states for higher confidence with interpretation!')
# Concating all results
df_results_100_generators = pd.concat(df_total_bootstrap, axis = 1)
# Extracting into csv file for results
df_results_100_generators.to_csv('/root/capsule/results/df_results_100_generators.csv')

# #### C) Benchmark Performance Visualization

# Production data outliers are rarely higher than 35% according to raters Q3 perforamnce in table 6
perc_list = np.arange(0.05,0.4, 0.05)
perc_columns = df_results_100_generators.columns.get_level_values(0).unique().tolist()[:-2]
# Averaging models performance across the entire bootstrapped dataset
df_outlying_mean = df_results_100_generators.groupby(df_results_100_generators.index).mean()[perc_columns]
# resetting index to be the same order 
df_outlying_mean = df_outlying_mean.reindex(df_results_100_generators.index.unique().tolist())
# converting to excel sheet
df_outlying_mean.to_excel('/root/capsule/results/Benchmark_results_on_100_randomly_generated_datasets.xlsx')
print('Calculated average results across the entire dataset!')

print('Visualizing algorithms performance across various outlying percentages!')
# Evaluation metrics for models evaluation on the bootstrapped dataset in same order
evaluation_metrics = ['maP', 'cohen_kappa', 'matthews_phi_coef', 'auc_score', 'f2_score', 'Fallout',
                      'Accuracy', 'Recall', 'Precision', 'execution_time']
fig = plt.figure(figsize = (25, 25))
for i, matrix in enumerate(evaluation_metrics):
    # Adding subplots in a loop
    ax = plt.subplot(4,3,i+1)
    # Characteristics of each plot
    plt.plot(perc_list, df_outlying_mean.loc[:, df_outlying_mean.columns.get_level_values(1) == matrix].loc['COMBO'].tolist(), '--', c = 'lightcoral', label = 'Ensemble')
    plt.plot(perc_list, df_outlying_mean.loc[:, df_outlying_mean.columns.get_level_values(1) == matrix].loc['ABOD'].tolist(), 'o', c = 'green', label = 'ABOD')
    plt.plot(perc_list, df_outlying_mean.loc[:, df_outlying_mean.columns.get_level_values(1) == matrix].loc['SOS'].tolist(), 's', c = 'blue', label = 'SOS')
    plt.plot(perc_list, df_outlying_mean.loc[:, df_outlying_mean.columns.get_level_values(1) == matrix].loc['KNN_median'].tolist(), 'D', c = 'magenta', label = 'KNN Median')
    plt.plot(perc_list, df_outlying_mean.loc[:, df_outlying_mean.columns.get_level_values(1) == matrix].loc['COF'].tolist(), '^', c = 'red', label = 'COF')
    plt.plot(perc_list, df_outlying_mean.loc[:, df_outlying_mean.columns.get_level_values(1) == matrix].loc['KNN'].tolist(), '>', c = 'orange', label = 'KNN')
    plt.plot(perc_list, df_outlying_mean.loc[:, df_outlying_mean.columns.get_level_values(1) == matrix].loc['iForest'].tolist(), '<', c = 'black', label = 'iForest')
    plt.plot(perc_list, df_outlying_mean.loc[:, df_outlying_mean.columns.get_level_values(1) == matrix].loc['LOF'].tolist(), 'v', c = 'lightseagreen', label = 'LOF')
    plt.yticks(np.arange(0, 1, 0.1));
    plt.xlabel('Outliers Percentage');
    plt.ylabel(matrix + ' score');
    plt.title(matrix);
    
# Title for the entire graph
plt.suptitle('Performance of different Models', fontsize = 15);
# Mutual Legend
handles, labels = ax.get_legend_handles_labels();
fig.legend(handles, labels, loc='right', handlelength=1, handleheight = 7, fontsize = 12);
# Saving figure
plt.savefig('/root/capsule/results/Benchmark_performance_on_100_randomly_generated_datasets.png')
print('Code run successfully!')
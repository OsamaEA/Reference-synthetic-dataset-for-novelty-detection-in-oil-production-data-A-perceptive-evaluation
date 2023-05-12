# Main librarires
import numpy as np
np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
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
# Full Alaska wells that would be used for clustering
df = pd.read_csv('.\inputs\Alaska_100_wells.csv', index_col = [0], header = [0,1])
# Re-setting dataframe 
df = df.drop(['Qo_cum_bbl', 'ProductionMethod', 'WellStatus'], axis = 1, level=1)
df.rename(columns = {'online_days_previous_month': 'month_online_days'}, inplace = True)
# Outliers days indices
outliers = pd.read_excel('.\inputs\wells_100_outliers_indices.xlsx')
# Fractions of outliers in each well
percentages = pd.read_excel('.\inputs\wells_100_outliers_fractions.xlsx')
percentages = percentages.set_index('well_name').T


# # 1. Data Wrangling
def clean_data_same(df, outliers_full_df, month_online_days = False):
    '''
        Change some data into outliers without adding new dataset as outliers
        INPUT:
              df: the original clean dataset with the following columns [Qo_bpd, online_days"if True"]
              outliers_sd_value: value of STD for white gaussian noise from each point
              perc_outliers: fraction of outliers in the dataset
              tolerance: the tolerance +/- below which an outlier would still be conisdered an inlier
                                                                                        '''
    # To get same output in random space everytime based on a given input
    np.random.seed(42)
    df_ready = []
    outliers_indices = []
    new_outlier_clean_values = []
    for i, well_name in enumerate(df.columns.get_level_values(0).unique().tolist()):
        df_trial = df[well_name]
        
        # Extracting outliers value from the given input-data
        df_outliers_raw = outliers_full_df[well_name]  
        # Analogy of outliers indices
        df_outliers_raw_indices = df_outliers_raw.dropna().tolist()
        # Making a copy of the dataset
        df_outliers = df_trial.copy()
        # outlying dataframe accordingly
        df_outliers = df_trial[df_trial.index.isin(df_outliers_raw_indices)] 
        # Remove nan or zero production values from the dataset
        df_outliers = df_outliers[df_outliers['Qo_bpd'].notna()]
        df_outliers = df_outliers[df_outliers['Qo_bpd'] != 0]
        # Same removal for original dataset
        df_trial = df_trial[df_trial['Qo_bpd'].notna()]
        df_trial = df_trial[df_trial['Qo_bpd'] != 0]    
        
        # Setting clean and outlier dataframes with their groudn truths
        df_clean = df_trial.copy()
        # setting ground truth to zero as they are originally inliers
        df_clean.loc[:, 'ground_truth'] = 0
        
        # Indices of outliers days
        df_outliers['ground_truth'] = 1
        outliers_index = sorted(df_outliers.index.tolist())
        # clean dataframes
        df_clean = df_clean[~df_clean.index.isin(outliers_index)]
        # Merged dataframe
        df_merge = pd.concat([df_clean, df_outliers], axis = 0)
        # Sorting index for the total dataframe now including inliers and outliers
        df_trial = df_merge.sort_index()

        # Replacing errors with nans as they would be data with no production available
        df_trial = df_trial.replace([np.inf, -np.inf, 0], np.nan)
        # Setting index values
        df_trial.index.names = ['days']
        # Replacing nan values of ground truth as zeros as we will measure against it (we will not consider them in evaluation)
        df_trial['ground_truth'] = df_trial['ground_truth'].replace([np.nan], 0)   
        
        # Calculating some variables that we might need to use later
        df_trial.loc[:, 'Qo_cum_bbl'] = df_trial.loc[:, 'Qo_bpd'].cumsum()  
        if month_online_days != False:
            df_trial.loc[:, 'online_days_cum'] = df_trial.loc[:, 'month_online_days'].cumsum()
        # Setting multi index columns
        df_trial.columns = pd.MultiIndex.from_product([[well_name], df_trial.columns])
        
        # Appending dataframes and indices
        df_ready.append(df_trial)
        outliers_indices.append(outliers_index)

    # Merging dataframes on idnex column so that we dont drop nan values by mistake
    df_new = reduce(lambda df_1,df_2 : pd.merge(df_1,df_2, how = 'outer', left_index = True, right_index = True), df_ready)
    
    return(df_new, outliers_indices)


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

def final_table(COMBO_matrix, ABOD_matrix, SOS_matrix, KNN_matrix_avg, LOF_matrix, IFor_matrix,
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
    
    return(df)

def compare_OD_methods(df_original, outliers_df_original, percentages, columns_to_include = 'all', x_col_plot = 'days', y_col_plot = 'Qo_bpd', scaling = True):
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
    df, outliers_original = clean_data_same(df_original, outliers_df_original)

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
        
        # Given outlying percentage
        perc = percentages[i]
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
        
    # Extracting final table of conusion matrix results
    df_final = final_table(COMBO_matrix, ABOD_matrix, SOS_matrix, KNN_matrix_avg, LOF_matrix, IFor_matrix,
                          COF_matrix, KNN_median_matrix)
    
    return(df_final)


# # 3. Results Extraction

# #### A) Results on real-field dataset that consists of 100 wells!
real_field_data_results = compare_OD_methods(df, outliers, percentages.iloc[0,:].tolist(), ['days', 'Qo_bpd'])
real_field_data_results.to_excel('.\outputs\experiment_2\real_field_data_results.xlsx')
real_field_data_results


# #### B) Results on Benchmark dataset at 20% outliers
# Loading the average results as reported in experiment#1
df_rfrnc_outly = pd.read_excel('.\outputs\experiment_3\Benchmark_results_on_100_randomly_generated_datasets.xlsx', header = [0,1], index_col = [0])
# results on 20% outliers fixed value only across the bootstrapped dataset
df_rfrnc_outly_20 = df_rfrnc_outly['0.2 Outliers']
df_rfrnc_outly_20


# #### C) Visualizin g Models Performance on both Datasets
# Evaluation metrics to compare against
eval_metrics = df_rfrnc_outly_20.columns.tolist()
fig = plt.figure(figsize = (20, 6))
# Algorithms to use in comparison
for i, algorithm in enumerate(['COMBO', 'ABOD']):
    for matrix in eval_metrics:
        # Setting x and y points for each plot
        x1, x2 = 0, 1
        y1, y2 = df_rfrnc_outly_20.loc[algorithm, matrix], real_field_data_results.loc[algorithm, matrix]
        # Connecting linear plots
        ax = plt.subplot(1,2, i+1)
        # showing the point and the line
        ax.plot([x1,x2],[y1,y2],'o-', label = matrix)
    # Setting labels
    plt.ylabel('{} Performance'.format(algorithm))
    plt.title('{} Performance on Real-Field Dataset and Reference-Dataset!'.format(algorithm));
    plt.xticks([x1,x2], ['Reference Dataset','Field Dataset']);
    # Sharing y-axis across both pltos
    plt.yticks(np.linspace(0,1,11))
# Title for the entire graph
plt.suptitle('Comparison of Algorithms Performance on Real dataset and Reference dataset!', fontsize =18);
# Mutual Legend
handles, labels = ax.get_legend_handles_labels();
fig.legend(handles, labels, loc='right', handlelength=1, handleheight = 3, fontsize = 10);
# Saving figure
plt.savefig('.\outputs\experiment_2\Algorithms_performance_on_real_and_reference_datasets.png')

# %%
"""
* Refactoring the DOCUWARE model (shared on 2025-06-06 by Chris Koetas) based on the materials:
* Snowflake Notebook: "Copy of FINAL_DOCUWARE_CHURN_MODEL_VERSION_FINAL_QA" (created on 250703 by EN)
# Used in Porduction:         "FINAL_DOCUWARE_CHURN_MODEL_VERSION_FINAL_UAT_2_CATBOOST.py" (created on 2025-06-27 by EN)
* The DOCUWARE model's notebook was updated by Justin Bovard on 250714 and 250812
-------------------------------------------------------------------------------
QUESTIONS:
cell135: Is this correctely commented out?
    # merged_5 = merged_4.merge(l1_cust_churned[["CUST_ACCOUNT_NUMBER", "CHURNED_FLAG", "CHURN_DATE"]], on = "CUST_ACCOUNT_NUMBER", how="inner")
"""
# %%
# -----------------------------------------------------------------------------
# Advisor: Justin Bovard
# email: Justin.Bovard@ricoh-usa.com
# -----------------------------------------------------------------------------
# Author: Evgeni Nikoolaev
# email: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# UPDATED ON: 2025-08-11
# CREATED ON: 2025-06-27
# -----------------------------------------------------------------------------
# COPYRIGHT@2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh
# -----------------------------------------------------------------------------
# %%
%load_ext autoreload
%autoreload 2
%load_ext watermark
%watermark
%matplotlib inline
%config InlineBackend.figure_format = 'retina'  # Optional: better quality
# %%
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100  # Optional: adjust figure size
# %% [markdown]
# # Import all required libraries
# Snowflake Cell: Importing_Libraries
import time
import math
import pandas as pd
import numpy as np
import re
import os
import sys
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from datetime import datetime
from snowflake.snowpark import Session
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import catboost as cb

from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit

from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series

from pathlib import Path
from churn_aiml.utils.profiling import timer
from churn_aiml.utils.device import check_torch_device
from churn_aiml.utils.find_paths import ProjectRootFinder
from churn_aiml.data.data_serializers.joblib_serializer import JoblibSerializer
from churn_aiml.ml.models.selection.train_test_split import StratifiedChurnSampler
from churn_aiml.ml.models.model_serializers.joblib_model_serializer import JoblibModelSerializer
from churn_aiml.visualization.evaluation_plots.prediction_panels import PredictionVisualizer
from churn_aiml.visualization.product_usage.docuware_usage_panels import UsageVisualizer

import warnings
warnings.filterwarnings("ignore")
# %%
print('Versions control:')
%watermark --iversions
# %%
_ = check_torch_device()
# %% [markdown]
# # Set paths to working folders
project_dir = ProjectRootFinder().find_path()
cwd = Path(__file__).parent
outputs = cwd / "outputs"
outputs.mkdir(exist_ok=True)
print(f"project_dir: {project_dir}")
print(f"cwd: {cwd}")
print(f"outputs: {outputs}")
# %%
write_to_outputs = True
read_from_outputs = True
# %% [markdown]
# # Set Snowflake session
# Snowflake Cell: Loading_Snowflake_Session
connection_parameters = {
            'user':'INTAIMLGEN',
            'password':'QPalzmbchdeul77##',
            'account':'ricohusa.us-east-1.privatelink',
            'warehouse':'RAC_AIML_NONPROD',
            'database':'RAC_RAPID_DEV',
}
session = Session.builder.configs(connection_parameters).create()
session.use_schema("RUS_AIML")
session.use_role("RAC_DEV_AIML_ROLE")
# %% [markdown]
# # UTILITY Function 1
# Snowflake Cell: cell4
def convert_to_float(df):
    return df.apply(pd.to_numeric, errors='coerce').astype(np.float32)
# %% [markdown]
# # UTILITY Function 2
# Snowflake Cell: cell1
def convert_yyyywk_to_date(yyyywk):
    number_of_days_in_month = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    time = str(yyyywk)
    year = int(time[:4])
    week_no = int(time[4:6])

    total_days = (week_no*7) - 3
    days_count = 0
    cummulative_sum = []
    cummulative_sum.append(0)

    for key,val in number_of_days_in_month.items():
        days_count += val
        if year%4 == 0 and key==2:
            days_count+=1
        cummulative_sum.append(days_count)

    first = 0
    second = 1
    month = 12

    while second < len(cummulative_sum):
        if (cummulative_sum[first] <= total_days) and (total_days <= cummulative_sum[second]):
            month = first+1
            day = total_days-cummulative_sum[first]
            break
        else:
            first+=1
            second+=1

    if month<10:
        month_date = str(year)+"-0"+str(month)+"-01"
    else:
        month_date = str(year)+"-"+str(month)+"-01"
    return month_date
# %%
# Snowflake Cell: cell78
def convert_yyyywk_to_actual_mid_date(yyyywk):
    number_of_days_in_month = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    time = str(yyyywk)
    year = int(time[:4])
    week_no = int(time[4:6])
    if week_no > 52:
        week_no = 52

    total_days = (week_no*7) - 3
    days_count = 0
    cummulative_sum = []
    cummulative_sum.append(0)

    for key,val in number_of_days_in_month.items():
        days_count += val
        if year%4 == 0 and key==2:
            days_count+=1
        cummulative_sum.append(days_count)

    first = 0
    second = 1
    month = 12
    day = 0
    while second < len(cummulative_sum):
        if (cummulative_sum[first] <= total_days) and (total_days <= cummulative_sum[second]):
            month = first+1
            day = total_days-cummulative_sum[first]
            break
        else:
            first+=1
            second+=1

    day_no="00"

    if day <10:
        day_no = "0"+str(day)
    else:
        day_no = str(day)

    if month<10:
        month_date = str(year)+"-0"+str(month)+"-"+day_no
    else:
        month_date = str(year)+"-"+str(month)+"-"+day_no
    return month_date

# %%
# Snowflake Cell: cell78
def get_days_diff(yyyywk, last_yyyywk):
    curr = pd.to_datetime(convert_yyyywk_to_actual_mid_date(yyyywk))
    last = pd.to_datetime(convert_yyyywk_to_actual_mid_date(last_yyyywk))
    #print(curr, last)
    days_diff = (last- curr) / np.timedelta64(1, 'D')
    return days_diff
# %%  [markdown]
get_days_diff(202318, 202401)
# %% [markdown]
# # UTILITY Function 3
# Snowflake Cell: cell2
def convert_date_to_yyyywk(d):

    number_of_days_in_month = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    date = pd.to_datetime(d)
    month = date.month
    year = date.year
    day = date.day
    week_number = 1
    days_count = 0
    cummulative_sum = []
    cummulative_sum.append(0)

    for key,val in number_of_days_in_month.items():
        days_count += val
        if year%4 == 0 and key==2:
            days_count+=1
        cummulative_sum.append(days_count)
    if month<13 and month>0:
        week_number = math.ceil((cummulative_sum[month-1]+day)/7)

    yyyywk="000000"
    if week_number>9:
        yyyywk = str(year)+str(week_number)
    else:
        yyyywk = str(year)+"0"+str(week_number)
    ans = int(yyyywk)
    return ans
# %% [markdown]
# # UTILITY Function 4
# Snowflake Cell: cell92
def engineer_timeseries_cols_using_tsfresh_for_live_customers(only_features):

    final_features = pd.DataFrame()

    # Store Dataframe into CSV files for each customer id and dates

    only_features['YYYYWK'] = only_features['YYYYWK'].astype(int)
    only_features['CUST_ACCOUNT_NUMBER'] = only_features['CUST_ACCOUNT_NUMBER'].astype(int)
    only_features = only_features.loc[:,~only_features.columns.str.contains('^Unnamed', case=False)]
    only_features = only_features.fillna(0.0)

    common_cust_id = only_features['CUST_ACCOUNT_NUMBER'].unique()
    start_time_begin = time.time()

    count=1
    for id_ in common_cust_id:
        print(count)
        start_time = time.time()
        flag=False

        print(id_)

        only_ts_df = only_features[only_features['CUST_ACCOUNT_NUMBER'] == id_]

        average_df = only_ts_df.groupby('YYYYWK')[['INVOICE_REVLINE_TOTAL', 'ORIGINAL_AMOUNT_DUE', 'FUNCTIONAL_AMOUNT', 'USED_STORAGE_MB', 'DOCUMENTS_OPENED']].sum()

        average_df.reset_index(inplace=True)
        average_df['CUST_ACCOUNT_NUMBER'] = id_

        average_df['YYYYWK'] = average_df['YYYYWK'].astype(int)
        average_df['CUST_ACCOUNT_NUMBER'] = average_df['CUST_ACCOUNT_NUMBER'].astype(int)

        #df_test = session.create_dataframe(average_df)
        #df_test.write.copy_into_location("@PS_DOCUWARE_CHURN/only_ts_df"+str(id_)+".csv", file_format_type="csv", format_type_options={"COMPRESSION": "NONE"}, header=True, overwrite=True )

        only_ts_df = average_df

        if only_ts_df.shape[0] > 1:
            df_rolled = roll_time_series(only_ts_df, column_id="CUST_ACCOUNT_NUMBER", column_sort="YYYYWK")          # roll_time_series
        elif only_ts_df.shape[0] == 1:
            flag=True
            df_rolled = only_ts_df
            df_rolled['id'] = "("+str(only_ts_df.loc[only_ts_df.index[0],'CUST_ACCOUNT_NUMBER'])+", "+str(only_ts_df.loc[only_ts_df.index[0], 'YYYYWK'])+")"
        else:
            continue

        columns_to_dropped = set()

        df_rolled_ = df_rolled.loc[:,~df_rolled.columns.str.contains('^(Unnamed|CUST_|YYYYWK|id)', case=False)]

        df_rolled_imputed = df_rolled_

        df_rolled_imputed['YYYYWK'] = df_rolled['YYYYWK'].to_list()
        df_rolled_imputed['ID'] = df_rolled['id'].to_list()
        df_rolled_imputed['CUST_ACCOUNT_NUMBER'] = df_rolled['CUST_ACCOUNT_NUMBER'].to_list()
        #df_rolled_imputed.rename(columns={'USED_STORAGE__MB':'USED_STORAGE_MB'}, errors="raise",inplace=True)

        columns_to_dropped = set()

        df_rolled_imputed = df_rolled_imputed.loc[:,~df_rolled_imputed.columns.str.contains('^Unnamed', case=False)]

        extraction_settings = ComprehensiveFCParameters()

        if df_rolled_imputed.shape[0] > 0:
            features = extract_features(df_rolled_imputed, column_id='ID', column_sort='YYYYWK', default_fc_parameters=extraction_settings, n_jobs=48) #extract_features

            df_temp = (features.sort_index().iloc[features.shape[0]-1, :]).to_frame().transpose()
            df_temp['CUST_ACCOUNT_NUMBER'] = id_

            final_features = pd.concat([final_features, df_temp], axis=0, ignore_index=True)

        del only_ts_df
        del features
        del df_rolled
        del df_rolled_
        del df_rolled_imputed

        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"for Cust ID = {id_} Elapsed time: {elapsed_time} seconds")
        count+=1
        #if count == 11:
        #    break
    end_time_begin = time.time()
    elapsed_time_begin = end_time_begin - start_time_begin
    #print(f"Total Elapsed time: {elapsed_time_begin} seconds")

    #Remove unwanted columns
    final_features_columns = final_features.columns
    columns_to_be_dropped_from_ts_df = []

    for col in final_features_columns:
        if re.search(r"^(Unnamed:|obr_|prtar|CONTRACT_LINE_ITEMS|PROBABILITY_OF_DELINQUENCY|RICOH_CUSTOM_RISK_MODEL|CUST_ACCOUNT_NUMBER_)", col):
            columns_to_be_dropped_from_ts_df.append(col)

    final_features_filtered = final_features.drop(columns_to_be_dropped_from_ts_df, axis=1)

    return final_features_filtered
# %% [markdown]
# # UTILITY Function 5
# Snowflake Cell: cell7
def engineer_timeseries_cols_using_tsfresh(only_features):

    final_features = pd.DataFrame()
    final_calculated = pd.DataFrame()
    # Store Dataframe into CSV files for each customer id and dates

    only_features['YYYYWK'] = only_features['YYYYWK'].astype(int)
    only_features['CUST_ACCOUNT_NUMBER'] = only_features['CUST_ACCOUNT_NUMBER'].astype(int)
    only_features = only_features.loc[:,~only_features.columns.str.contains('^Unnamed', case=False)]
    only_features = only_features.fillna(0.0)

    common_cust_id = only_features['CUST_ACCOUNT_NUMBER'].unique()
    start_time_begin = time.time()

    count=1
    for id_ in common_cust_id:
        print(count)
        start_time = time.time()
        flag=False

        print(id_)
        calculated = pd.DataFrame(columns=['DAYS_REMAINING'])
        only_ts_df = only_features[only_features['CUST_ACCOUNT_NUMBER'] == id_]
        filename = "@PS_DOCUWARE_CHURN/TSFRESH/all_feature_TSFRESH/only_ts_df_"+str(id_)+".csv"
        curr_df = session.create_dataframe(only_ts_df)
        #curr_df.write.copy_into_location("@PS_DOCUWARE_CHURN/TSFRESH/all_feature_TSFRESH/only_ts_df"+str(id_)+".csv", file_format_type="csv", format_type_options={"COMPRESSION": "NONE"}, header=True, overwrite=True )
        #curr_df.write.csv(filename, overwrite=True, single=True)
        #session.file.put(only_ts_df, ,overwrite=True)


        #print( calculated.shape )
        sum_df = only_ts_df.groupby('YYYYWK')[['INVOICE_REVLINE_TOTAL', 'ORIGINAL_AMOUNT_DUE', 'FUNCTIONAL_AMOUNT', 'USED_STORAGE_MB', 'DOCUMENTS_OPENED']].sum()

        sum_df.reset_index(inplace=True)
        sum_df['CUST_ACCOUNT_NUMBER'] = id_

        sum_df['YYYYWK'] = sum_df['YYYYWK'].astype(int)
        sum_df['CUST_ACCOUNT_NUMBER'] = sum_df['CUST_ACCOUNT_NUMBER'].astype(int)

        #df_test = session.create_dataframe(sum_df)
        #df_test.write.copy_into_location("@PS_DOCUWARE_CHURN/only_ts_df"+str(id_)+".csv", file_format_type="csv", format_type_options={"COMPRESSION": "NONE"}, header=True, overwrite=True )

        only_ts_df = sum_df
        df_test = session.create_dataframe(sum_df)
        #df_test.write.copy_into_location("@PS_DOCUWARE_CHURN/TSFRESH/all_feature_TSFRESH/sum_ts_df"+str(id_)+".csv", file_format_type="csv", format_type_options={"COMPRESSION": "NONE"}, header=True, overwrite=True )
        #print("1", only_ts_df.shape)

        calculated['DAYS_REMAINING'] = only_ts_df['YYYYWK'].apply(get_days_diff, args=(only_ts_df.loc[only_ts_df.index[only_ts_df.shape[0]-1], 'YYYYWK'],))
        # print("-------------------------------")
        # print(calculated['DAYS_REMAINING'].shape)
        # print("***********")
        # print(calculated['DAYS_REMAINING'])
        # print("#####################")
        if only_ts_df.shape[0] > 1:
            df_rolled = roll_time_series(only_ts_df, column_id="CUST_ACCOUNT_NUMBER", column_sort="YYYYWK")          # roll_time_series
        elif only_ts_df.shape[0] == 1:
            flag=True
            df_rolled = only_ts_df
            df_rolled['id'] = "("+str(only_ts_df.loc[only_ts_df.index[0],'CUST_ACCOUNT_NUMBER'])+", "+str(only_ts_df.loc[only_ts_df.index[0], 'YYYYWK'])+")"
        else:
            continue

        columns_to_dropped = set()

        df_rolled_ = df_rolled.loc[:,~df_rolled.columns.str.contains('^(Unnamed|CUST_|YYYYWK|id)', case=False)]

        df_rolled_imputed = df_rolled_

        df_rolled_imputed['YYYYWK'] = df_rolled['YYYYWK'].to_list()
        df_rolled_imputed['ID'] = df_rolled['id'].to_list()
        df_rolled_imputed['CUST_ACCOUNT_NUMBER'] = df_rolled['CUST_ACCOUNT_NUMBER'].to_list()
        #df_rolled_imputed.rename(columns={'USED_STORAGE__MB':'USED_STORAGE_MB'}, errors="raise",inplace=True)

        columns_to_dropped = set()

        df_rolled_imputed = df_rolled_imputed.loc[:,~df_rolled_imputed.columns.str.contains('^Unnamed', case=False)]

        extraction_settings = ComprehensiveFCParameters()

        if df_rolled_imputed.shape[0] > 0:
            features = extract_features(df_rolled_imputed, column_id='ID', column_sort='YYYYWK', default_fc_parameters=extraction_settings, n_jobs=48) #extract_features
            print("2",features.shape)
            #df_temp = (features.sort_index().iloc[features.shape[0]-1, :]).to_frame().transpose()
            #df_temp = (features.sort_index().loc[:, :]).transpose()
            #print(calculated['DAYS_REMAINING'])
            features['CUST_ACCOUNT_NUMBER'] = id_
            features = features.reset_index(drop=True)
            print("3",features.shape)

            #features_df = session.create_dataframe(features)
            #features_df.write.copy_into_location("@PS_DOCUWARE_CHURN/TSFRESH/all_feature_TSFRESH/features_df"+str(id_)+".csv", file_format_type="csv", format_type_options={"COMPRESSION": "NONE"}, header=True, overwrite=True )


            #features_with_age = pd.concat([features, calculated], axis=1, ignore_index=True)

            #features_with_age_df = session.create_dataframe(features_with_age)
            #features_with_age_df.write.copy_into_location("@PS_DOCUWARE_CHURN/TSFRESH/all_feature_TSFRESH/features_with_age_df"+str(id_)+".csv", file_format_type="csv", format_type_options={"COMPRESSION": "NONE"}, header=True, overwrite=True )

            print("4",features.shape)
            final_features = pd.concat([final_features, features], axis=0, ignore_index=True)
            final_calculated = pd.concat([final_calculated, calculated], axis=0, ignore_index=True)
            #print(calculated['DAYS_REMAINING'])

            #final_features_df = session.create_dataframe(final_features)
            #final_features_df.write.copy_into_location("@PS_DOCUWARE_CHURN/TSFRESH/all_feature_TSFRESH/final_features_df"+str(id_)+".csv", file_format_type="csv", format_type_options={"COMPRESSION": "NONE"}, header=True, overwrite=True )

            print("5", final_features.shape)
            print("6", final_calculated.shape)

        del only_ts_df
        del features
        del df_rolled
        del df_rolled_
        del df_rolled_imputed
        del calculated

        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"for Cust ID = {id_} Elapsed time: {elapsed_time} seconds")
        count+=1
        #if count==5:
        #    break
    end_time_begin = time.time()
    elapsed_time_begin = end_time_begin - start_time_begin
    #print(f"Total Elapsed time: {elapsed_time_begin} seconds")

    #Remove unwanted columns
    final_features_columns = final_features.columns
    columns_to_be_dropped_from_ts_df = []
    #print(final_features.columns)
    for col in final_features_columns:
        if re.search(r"^(Unnamed:|obr_|prtar|CONTRACT_LINE_ITEMS|PROBABILITY_OF_DELINQUENCY|RICOH_CUSTOM_RISK_MODEL|CUST_ACCOUNT_NUMBER_)", col):
            columns_to_be_dropped_from_ts_df.append(col)

    final_features_filtered = final_features.drop(columns_to_be_dropped_from_ts_df, axis=1)

    return final_features_filtered, final_calculated

# %%  [markdown]
# # UTILITY Function 6
# Snowflake Cell: cell30
def global_shap_importance(model, X):
    """ Return a dataframe containing the features sorted by Shap importance
    Parameters
    ----------
    model : The tree-based model
    X : pd.Dataframe
         training set/test set/the whole dataset ... (without the label)
    Returns
    -------
    pd.Dataframe
        A dataframe containing the features sorted by Shap importance
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    cohorts = {"": shap_values}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
    feature_importance = pd.DataFrame(
        list(zip(feature_names, sum(values))), columns=['features', 'importance'])
    feature_importance.sort_values(
        by=['importance'], ascending=False, inplace=True)
    return feature_importance
# %% [markdown]
# # UTILITY FUNCTION 7
# Snowflake Cell: cell133
def sort_contract_line_items(row):
    arr = row.split(',')
    cell_val = ""
    arr_1 = sorted(arr)

    if len(arr_1)>0:
        cell_val = arr_1[0].strip()
    for elem in arr_1:
        cell_val = cell_val+","+elem.strip()
    return cell_val
# %% [markdown]
# # Raw Data Extraction
# # Edited by Justin Bovard on 2025-07-14
# Snowflake Cell: Impute_Missing_Usage_23
# Code to impute missing usage data for first half of 2023
with timer():
    usage_latest = session.sql("""
        WITH latback AS (
            SELECT DISTINCT CONTRACT_NUMBER,
                REGEXP_REPLACE(trim(CUSTOMER_NAME), '  ', ' ') AS CUSTOMER_NAME,
                CONTRACT_START,
                CONTRACT_END,
                DOCUMENTS_OPENED,
                USED_STORAGE__MB,
                CONTRACT_LINE_ITEMS,
                PERIOD,
                YYYYWK
            FROM RUS_AIML.DOCUWARE_USAGE_COMBINED_V
        ),
        jaro AS (
            SELECT DISTINCT a.CUST_ACCOUNT_NUMBER,
                a.churn_date,
                a.churned_flag,
                b.CONTRACT_NUMBER,
                a.CUST_PARTY_NAME,
                b.CUSTOMER_NAME,
                b.CONTRACT_START,
                b.CONTRACT_END,
                b.DOCUMENTS_OPENED,
                b.USED_STORAGE__MB,
                b.CONTRACT_LINE_ITEMS,
                b.PERIOD,
                b.YYYYWK,
                jarowinkler_similarity(a.CUST_PARTY_NAME, b.CUSTOMER_NAME),
                rank() OVER (
                    PARTITION BY a.CUST_ACCOUNT_NUMBER
                    ORDER BY jarowinkler_similarity(a.CUST_PARTY_NAME, b.CUSTOMER_NAME) DESC
                ) AS match_rank
            FROM RUS_AIML.PS_DOCUWARE_CUST_FINAL a
            FULL OUTER JOIN latback b
            WHERE jarowinkler_similarity(a.CUST_PARTY_NAME, b.CUSTOMER_NAME) BETWEEN 93 AND 100
        )
        SELECT *
        FROM jaro
        WHERE jaro.match_rank = 1
    """).to_pandas()
# %%
# # The foloiwng line is found in the old code developed by Chris Koetas
# usage_latest = session.sql("SELECT * FROM RUS_AIML.DOCUWARE_USAGE_JAPAN_V1_LATEST_V").to_pandas()
usage_latest['CONTRACT_START'] = pd.to_datetime(usage_latest['CONTRACT_START'])
usage_latest['CHURN_DATE']= pd.to_datetime(usage_latest['CHURN_DATE'])
usage_latest['YYYYWK'] = usage_latest['YYYYWK'].astype('Int64')

# Focus only on Contract Start Date >= ‘2020-01-01’
usageFIX = usage_latest[usage_latest['CONTRACT_START']>= '2020-01-01' ]

# Active -  If contract start date <= 2022-12-31  for active then we need to add weekly usage numbers ( average of last 10 for numerical)
usageFIXActive = usageFIX[(usageFIX['CONTRACT_START']<= '2022-12-31') & (usageFIX['CHURNED_FLAG']== 'N') & (usageFIX['YYYYWK'] <= 202252) ]
idx = usageFIXActive.groupby('CUST_ACCOUNT_NUMBER')['YYYYWK'].idxmax()
max_scores = usageFIXActive.loc[idx]
df_duplicated = pd.DataFrame(np.repeat(max_scores.values, repeats=25, axis=0), columns=max_scores.columns)
df_duplicated= df_duplicated.sort_values(by = 'CUST_ACCOUNT_NUMBER')

# range of YYYYWK missing data
recurring_range = []
for i in range(1,26):

    if i < 10:
        month_date = str(2023)+"0"+str(i)
    else:
        month_date = str(2023)+str(i)
    recurring_range.append(month_date)

# Calculate how many times the range needs to repeat
num_repeats = len(max_scores)    #len(df) // len(recurring_range) + 1

# Create the new column values by repeating the range and truncating
new_column = np.tile(recurring_range, num_repeats)[:len(df_duplicated)]

# Replace the column
df_duplicated['YYYYWK'] = new_column

# Active account imputations
ActiveAccts = df_duplicated

# Churned Customers part 1    If contract start date <= 2022-12-31  and churned date after > 2022-12-31 then add in usage for wks between 202301-202325
usageFIXChurned1 = usageFIX[(usageFIX['CONTRACT_START']<= '2022-12-31') & (usageFIX['CHURNED_FLAG']== 'Y') &  (usageFIX['CHURN_DATE'] > '2023-06-24' ) & (usageFIX['YYYYWK'] <= 202252)   ]

idx = usageFIXChurned1.groupby('CUST_ACCOUNT_NUMBER')['YYYYWK'].idxmax()
max_scores = usageFIXChurned1.loc[idx]
df_duplicated = pd.DataFrame(np.repeat(max_scores.values, repeats=25, axis=0), columns=max_scores.columns)
df_duplicated.sort_values(by = 'CUST_ACCOUNT_NUMBER')

# Calculate how many times the range needs to repeat
num_repeats = len(max_scores)   #len(df) // len(recurring_range) + 1

# Create the new column values by repeating the range and truncating
new_column = np.tile(recurring_range, num_repeats)[:len(df_duplicated)]

# Replace the column
df_duplicated['YYYYWK'] = new_column

# save imputations for part 1 of churned customers
churned1 = df_duplicated

# end churned customer part1

# Churned Customers part 2 those customers who churned during the data outage
usageFIXChurned2 = usageFIX[(usageFIX['CONTRACT_START']<= '2022-12-31') & (usageFIX['CHURNED_FLAG']== 'Y') &  (usageFIX['CHURN_DATE'] <= '2023-06-24' ) & (usageFIX['YYYYWK'] <= 202252) & (usageFIX['CHURN_DATE'] >= '2023-01-01' )  ]

idx = usageFIXChurned2.groupby('CUST_ACCOUNT_NUMBER')['YYYYWK'].idxmax()
max_scores = usageFIXChurned2.loc[idx]
max_scores['ywk'] = [(i -pd.to_datetime('2023-01-01')).days//7 for i in max_scores['CHURN_DATE']]

# Method to duplicate rows
df_duplicated = max_scores.loc[np.repeat(max_scores.index.values, max_scores['ywk'])]

# Reset index if needed
df_duplicated = df_duplicated.reset_index(drop=True)

# Replace hard coded range
start_index = 0
end_index = 23
new_values = recurring_range[0:24]

df_duplicated.loc[start_index:end_index, 'YYYYWK'] = new_values

#  Replace hard coded range
start_index = 24
end_index = 46
new_values = recurring_range[0:23]

df_duplicated.loc[start_index:end_index, 'YYYYWK'] = new_values

#  Replace hard coded range
start_index = 47
end_index = 56
new_values = recurring_range[0:10]

df_duplicated.loc[start_index:end_index, 'YYYYWK'] = new_values

# Replace hard coded range
start_index = 57
end_index = 80
new_values = recurring_range[0:24]

df_duplicated.loc[start_index:end_index, 'YYYYWK'] = new_values

# Replace hard coded range
start_index = 81
end_index = 87
new_values = recurring_range[0:7]

df_duplicated.loc[start_index:end_index, 'YYYYWK'] = new_values

# save imputations for part 2 of churned customers
churned2 = df_duplicated

#Combine imputations
imputations = pd.concat([ActiveAccts,churned1,churned2])
# drop added columns from imputations
imputations = imputations.drop(columns=['ywk']) #'Unnamed: 0' in snapshot
usage = pd.concat([usageFIX,imputations])
usage_latest = usage
print(f"usage_latest.shape: {usage_latest.shape}")
print(usage_latest.head().to_string())
# %%
print(usage['YYYYWK'])
# %% [markdown]
# # Preprocessing Raw Data
# Snowflake Cell: Sproc_2
# Justin Bovard 2025-07-14
pay_df = session.sql("SELECT CUSTOMER_NO, RECEIPT_DATE, FUNCTIONAL_AMOUNT FROM RUS_AIML.PS_DOCUWARE_PAYMENTS").to_pandas()
rev_df = session.sql("SELECT CUST_ACCOUNT_NUMBER, DATE_INVOICE_GL_DATE, INVOICE_REVLINE_TOTAL FROM RUS_AIML.PS_DOCUWARE_REVENUE").to_pandas()
trx_df = session.sql("SELECT ACCOUNT_NUMBER, TRX_DATE, ORIGINAL_AMOUNT_DUE FROM RUS_AIML.PS_DOCUWARE_TRX").to_pandas()
contracts_sub_df = session.sql("SELECT CUST_ACCOUNT_NUMBER, SLINE_START_DATE, SLINE_END_DATE, SLINE_STATUS FROM RUS_AIML.PS_DOCUWARE_CONTRACT_SUBLINE").to_pandas()
renewals_df = session.sql("SELECT TO_CHAR(BILLTOCUSTOMERNUMBER) AS BILLTOCUSTOMERNUMBER, TO_CHAR(SHIPTOCUSTNUM) AS SHIPTOCUSTNUM, STARTDATECOVERAGE, CONTRACT_END_DATE FROM RUS_AIML.PS_DOCUWARE_SSCD_RENEWALS").to_pandas()
l1_cust_df =  session.sql("SELECT CUST_ACCOUNT_NUMBER, CUST_PARTY_NAME, L3_RISE_CONSOLIDATED_NUMBER, L3_RISE_CONSOLIDATED_NAME, L2_RISE_CONSOLIDATED_NUMBER, L2_RISE_CONSOLIDATED_NAME, CUST_ACCOUNT_TYPE, CUSTOMER_SEGMENT, CUSTOMER_SEGMENT_LEVEL, CHURNED_FLAG, CHURN_DATE FROM RUS_AIML.PS_DOCUWARE_L1_CUST").to_pandas()
dnb_risk_df = session.sql("SELECT TO_CHAR(ACCOUNT_NUMBER) AS ACCOUNT_NUMBER, OVERALL_BUSINESS_RISK, RICOH_CUSTOM_RISK_MODEL, PROBABILITY_OF_DELINQUENCY, PAYMENT_RISK_TRIPLE_A_RATING FROM RUS_AIML.DNB_RISK_BREAKDOWN").to_pandas()
# usage_latest = session.sql("SELECT * FROM RUS_AIML.DOCUWARE_USAGE_JAPAN_V1_LATEST_V").to_pandas()

# Removing duplicates from all the pandas dataframes
pay_df = pay_df.drop_duplicates()
rev_df = rev_df.drop_duplicates()
trx_df = trx_df.drop_duplicates()
contracts_sub_df = contracts_sub_df.drop_duplicates()
renewals_df = renewals_df.drop_duplicates()
l1_cust_df = l1_cust_df.drop_duplicates()
dnb_risk_df = dnb_risk_df.drop_duplicates()
usage_latest = usage_latest.drop_duplicates()

# Selecting only churned customers

l1_cust_churned = l1_cust_df[l1_cust_df["CHURNED_FLAG"]=='Y']

pay_df_churned = pay_df.merge(l1_cust_churned[["CUST_ACCOUNT_NUMBER"]], left_on = "CUSTOMER_NO", right_on = "CUST_ACCOUNT_NUMBER", how="inner")
pay_df_churned = pay_df_churned.drop("CUSTOMER_NO", axis=1)
pay_df_churned["RECEIPT_DATE"] = pd.to_datetime(pay_df_churned["RECEIPT_DATE"])
pay_df_churned = pay_df_churned.groupby(["CUST_ACCOUNT_NUMBER", "RECEIPT_DATE"]).agg('sum').reset_index()
pay_df_churned["MONTH"] = pd.to_datetime(pay_df_churned["RECEIPT_DATE"])

rev_df_churned = rev_df.merge(l1_cust_churned[["CUST_ACCOUNT_NUMBER"]], on = "CUST_ACCOUNT_NUMBER", how="inner")
rev_df_churned["DATE_INVOICE_GL_DATE"] = pd.to_datetime(rev_df_churned["DATE_INVOICE_GL_DATE"])
rev_df_churned = rev_df_churned.groupby(["CUST_ACCOUNT_NUMBER", "DATE_INVOICE_GL_DATE"]).agg('sum').reset_index()
rev_df_churned["MONTH"] = pd.to_datetime(rev_df_churned["DATE_INVOICE_GL_DATE"])

trx_df_churned = trx_df.merge(l1_cust_churned[["CUST_ACCOUNT_NUMBER"]], left_on = "ACCOUNT_NUMBER", right_on = "CUST_ACCOUNT_NUMBER", how="inner")
trx_df_churned = trx_df_churned.drop("ACCOUNT_NUMBER", axis=1)
trx_df_churned["TRX_DATE"] = pd.to_datetime(trx_df_churned["TRX_DATE"])
trx_df_churned = trx_df_churned.groupby(["CUST_ACCOUNT_NUMBER", "TRX_DATE"]).agg('sum').reset_index()
trx_df_churned["MONTH"] = pd.to_datetime(trx_df_churned["TRX_DATE"])

p_r_t_merged = pay_df_churned.merge(rev_df_churned, on = ["CUST_ACCOUNT_NUMBER","MONTH"], how="outer").merge(trx_df_churned, on = ["CUST_ACCOUNT_NUMBER","MONTH"], how="outer")

#Contracts Subline
contracts_sub_df["SLINE_START_DATE"] = pd.to_datetime(contracts_sub_df["SLINE_START_DATE"])
contracts_sub_df["SLINE_END_DATE"] = pd.to_datetime(contracts_sub_df["SLINE_END_DATE"])

contracts_sub_df["SUB_EARLIEST_DATE"] = contracts_sub_df.groupby("CUST_ACCOUNT_NUMBER")["SLINE_START_DATE"].transform("min")
contracts_sub_df["SUB_LATEST_DATE"] = contracts_sub_df.groupby("CUST_ACCOUNT_NUMBER")["SLINE_END_DATE"].transform("max")
contracts_sub_df_churned = contracts_sub_df.merge(l1_cust_churned[["CUST_ACCOUNT_NUMBER"]], on = "CUST_ACCOUNT_NUMBER", how="inner")

contracts_sub_df_churned = contracts_sub_df_churned.drop_duplicates()

# Renewals
cols_to_str = ["BILLTOCUSTOMERNUMBER","SHIPTOCUSTNUM"]
renewals_df[cols_to_str] = renewals_df[cols_to_str].astype("Int64").astype(str)

renewals_df_churned_1 = renewals_df.merge(l1_cust_churned[["CUST_ACCOUNT_NUMBER"]], left_on = "SHIPTOCUSTNUM", right_on = "CUST_ACCOUNT_NUMBER",  how="left")

renewals_df_churned_2 = renewals_df_churned_1.merge(l1_cust_churned[["CUST_ACCOUNT_NUMBER"]], left_on = "BILLTOCUSTOMERNUMBER", right_on = "CUST_ACCOUNT_NUMBER",  how="left", suffixes = ("_1", "_2"))
renewals_df_churned_2["CUST_ACCOUNT_NUMBER"] = renewals_df_churned_2["CUST_ACCOUNT_NUMBER_1"].fillna(renewals_df_churned_2["CUST_ACCOUNT_NUMBER_2"])
renewals_df_churned_2 = renewals_df_churned_2.drop(["BILLTOCUSTOMERNUMBER", "SHIPTOCUSTNUM","CUST_ACCOUNT_NUMBER_1","CUST_ACCOUNT_NUMBER_2"], axis=1)
renewals_df_churned_2 = renewals_df_churned_2.dropna()

renewals_df_churned_2["STARTDATECOVERAGE"] =  pd.to_datetime(renewals_df_churned_2["STARTDATECOVERAGE"])
renewals_df_churned_2["RENEWALS_EARLIEST_DATE"] = renewals_df_churned_2.groupby(renewals_df_churned_2["CUST_ACCOUNT_NUMBER"])["STARTDATECOVERAGE"].transform("min")
renewals_df_churned_2["RENEWALS_LATEST_DATE"] = renewals_df_churned_2.groupby(renewals_df_churned_2["CUST_ACCOUNT_NUMBER"])["CONTRACT_END_DATE"].transform("max")

#DNB Risk Breakdown
dnb_risk_df["ACCOUNT_NUMBER"] = dnb_risk_df["ACCOUNT_NUMBER"].astype("Int64").astype(str)
dnb_risk_df_churned = dnb_risk_df.merge(l1_cust_churned[["CUST_ACCOUNT_NUMBER"]], left_on = "ACCOUNT_NUMBER", right_on = "CUST_ACCOUNT_NUMBER",  how="inner")
dnb_risk_df_churned = dnb_risk_df_churned.drop("ACCOUNT_NUMBER", axis=1)

usage_latest["YYYYWK_Transformed"] = pd.to_datetime(usage_latest["YYYYWK"].apply(convert_yyyywk_to_actual_mid_date), errors = "coerce")

#Usage Japan Latest
usage_latest["CUST_ACCOUNT_NUMBER"] = usage_latest["CUST_ACCOUNT_NUMBER"].astype("Int64").astype(str)
usage_latest["YYYYWK_MONTH"] = pd.to_datetime(usage_latest["YYYYWK_Transformed"])

# Adding Aggregate syntax using Group by on YYYYWK_MONTH
usage_latest['CONTRACT_LINE_ITEMS_CANONICAL_FORM'] = usage_latest['CONTRACT_LINE_ITEMS'].apply(sort_contract_line_items)
usage_latest.drop(['PERIOD', 'CONTRACT_LINE_ITEMS'], inplace=True, axis=1)
usage_latest = usage_latest.drop_duplicates()
df_full_final = usage_latest.dropna(subset=['CONTRACT_LINE_ITEMS_CANONICAL_FORM']).assign(something=lambda x:x['CONTRACT_LINE_ITEMS_CANONICAL_FORM'].str.len()).sort_values(['CUST_ACCOUNT_NUMBER','YYYYWK','something'], ascending=[True, True, False]).groupby(['CUST_ACCOUNT_NUMBER', 'YYYYWK'], as_index=False).head(1)
usage_latest = df_full_final.drop('something', axis=1)
usage_latest = usage_latest.rename(columns={'CONTRACT_LINE_ITEMS_CANONICAL_FORM':'CONTRACT_LINE_ITEMS'})
# %% END of Sproc_2
usage_latest["YYYYWK_Transformed"] = pd.to_datetime(usage_latest["YYYYWK"].apply(convert_yyyywk_to_actual_mid_date), errors = "coerce")
# %%
print(usage_latest["YYYYWK_Transformed"])
# %%
# Snowflake Cell: cell138
usage_latest_churned = usage_latest.merge(l1_cust_churned[["CUST_ACCOUNT_NUMBER"]], on="CUST_ACCOUNT_NUMBER",  how="inner")
# Merging all the churned customers data frames i.e. Payments, Revenue, Transactions, contracts, contracts subline, contracts topline, renewals, snow inc, tech survey, loyalty survey, dnb risk and usage latest
merged_1 = p_r_t_merged.merge(contracts_sub_df_churned, left_on = ["CUST_ACCOUNT_NUMBER", "MONTH"], right_on = ["CUST_ACCOUNT_NUMBER", "SLINE_START_DATE"], how="outer")
merged_1["MONTH"] = merged_1["MONTH"].fillna(merged_1["SLINE_START_DATE"])
print(f"merged_1.shape: {merged_1.shape}")
print(merged_1.head().to_string())
print()
merged_2 = merged_1.merge(renewals_df_churned_2, left_on = ["CUST_ACCOUNT_NUMBER","MONTH"], right_on = ["CUST_ACCOUNT_NUMBER","STARTDATECOVERAGE"], how="outer")
merged_2["MONTH"] = merged_2["MONTH"].fillna(merged_2["STARTDATECOVERAGE"])
print(f"merged_2.shape: {merged_2.shape}")
print(merged_2.head().to_string())
# %%
print(usage_latest_churned["YYYYWK_MONTH"].unique())
# %%
print(merged_2["STARTDATECOVERAGE"].unique())
# %%
# Snowflake Cell: cell135
merged_3 = merged_2.merge(dnb_risk_df_churned, on="CUST_ACCOUNT_NUMBER", how="left")

merged_4 = merged_3.merge(usage_latest_churned, left_on= ["CUST_ACCOUNT_NUMBER","MONTH"], right_on = ["CUST_ACCOUNT_NUMBER", "YYYYWK_MONTH"], how="outer")
merged_4["MONTH"] = merged_4["MONTH"].fillna(merged_4["YYYYWK_MONTH"])

to_drop_2 = ["CONTRACT_NUMBER", "CUST_PARTY_NAME", "CUSTOMER_NAME", "CONTRACT_END","JAROWINKLER_SIMILARITY(A.CUST_PARTY_NAME, B.CUSTOMER_NAME)"]

merged_5 = merged_4.drop(to_drop_2, axis=1)

# merged_5 = merged_4.merge(l1_cust_churned[["CUST_ACCOUNT_NUMBER", "CHURNED_FLAG", "CHURN_DATE"]], on = "CUST_ACCOUNT_NUMBER", how="inner")

merged_5["EARLIEST_DATE"] = merged_5[["RECEIPT_DATE", "DATE_INVOICE_GL_DATE", "TRX_DATE", "SLINE_START_DATE", "STARTDATECOVERAGE"]].min(axis=1)
merged_5["FINAL_EARLIEST_DATE"] = merged_5.groupby("CUST_ACCOUNT_NUMBER")["EARLIEST_DATE"].transform("min")

merged_5["CHURN_DATE"] = pd.to_datetime(merged_5["CHURN_DATE"])
merged_5["CHURN_MONTH"] = pd.to_datetime(merged_5["CHURN_DATE"]).dt.to_period("M").dt.to_timestamp()


merged_5["LIFESPAN_MONTHS"] = ((merged_5["CHURN_DATE"] - merged_5["FINAL_EARLIEST_DATE"]).dt.days) / 30
merged_5["DAYS_TO_CHURN"]  = ((merged_5["CHURN_DATE"] - merged_5["FINAL_EARLIEST_DATE"]).dt.days)

to_drop_3 = ["SLINE_END_DATE", "SLINE_STATUS", "SUB_EARLIEST_DATE", "SUB_LATEST_DATE", "RENEWALS_EARLIEST_DATE", "RENEWALS_LATEST_DATE", "CONTRACT_END_DATE", "CHURNED_FLAG","CHURN_MONTH", "EARLIEST_DATE"]

merged_5 = merged_5.drop(to_drop_3, axis=1)
# %%
print(merged_3[['CUST_ACCOUNT_NUMBER', 'MONTH']])
# %%
print(usage_latest_churned[['CUST_ACCOUNT_NUMBER', 'YYYYWK_MONTH']])
# %%
print(merged_3['CUST_ACCOUNT_NUMBER'].nunique())
# %%
print(usage_latest_churned['CUST_ACCOUNT_NUMBER'].nunique())
# %%
print(merged_4['YYYYWK'])
# %%
print(f"len(merged_5.columns) = {len(merged_5.columns)}")
print("\n".join(list(merged_5.columns)))
# %%
print(merged_5["STARTDATECOVERAGE"].unique())
# %%
print(merged_5["YYYYWK_MONTH"].unique())
# %%
print(merged_5["DATE_INVOICE_GL_DATE"].unique())
# %%
date_col = ["MONTH","YYYYWK_Transformed", "FINAL_EARLIEST_DATE", "CHURN_DATE"]
merged_5[date_col] = merged_5[date_col].astype(str)
# %%
merged_5 = merged_5.drop_duplicates()
# %%
print(f"merged_5.shape: {merged_5.shape}")
print(merged_5.head().to_string())
# %%
# usage_latest_churned['CUST_ACCOUNT_NUMBER'].unique()
# Snowflake Cell: cell148
temp = pd.DataFrame({'CUST_ACCOUNT_NUMBER':usage_latest_churned['CUST_ACCOUNT_NUMBER'].unique()})
merge_final = merged_5.merge(temp, on='CUST_ACCOUNT_NUMBER', how='inner')
# %%
print(merge_final["DATE_INVOICE_GL_DATE"].unique())
# %%
print(merge_final["STARTDATECOVERAGE"].unique())
# %%
print(merge_final["YYYYWK_MONTH"].unique())
# %%
print(merge_final["RECEIPT_DATE"].unique())
# %%
merge_final["RECEIPT_DATE"]= pd.to_datetime(merge_final["RECEIPT_DATE"])
merge_final["YYYYWK_MONTH"]= pd.to_datetime(merge_final["YYYYWK_MONTH"])
merge_final["DATE_INVOICE_GL_DATE"] = pd.to_datetime(merge_final["DATE_INVOICE_GL_DATE"])
# %%
print(merge_final["DATE_INVOICE_GL_DATE"].head())
# %%
# Snowflake Cell: cell160
def timestamp_To_Date(row):
    if type(row) == pd._libs.tslibs.nattype.NaTType:
        return None
    else:
        return pd.Timestamp(row).strftime('%Y-%m-%d')
# %%
merge_final["DATE_INVOICE_GL_DATE_CANONICAL"] = merge_final["DATE_INVOICE_GL_DATE"].apply(timestamp_To_Date)
print(merge_final["DATE_INVOICE_GL_DATE_CANONICAL"].head().to_string())
# %%
print(merge_final["RECEIPT_DATE"].head().to_string())
print(merge_final["RECEIPT_DATE"].tail().to_string())
# %%
merge_final["RECEIPT_DATE_CANONICAL"] = merge_final["RECEIPT_DATE"].apply(timestamp_To_Date)
print(merge_final["RECEIPT_DATE_CANONICAL"].head().to_string())
print(merge_final["RECEIPT_DATE_CANONICAL"].tail().to_string())
# %%
merge_final["YYYYWK_MONTH_CANONICAL"] = merge_final["YYYYWK_MONTH"].apply(timestamp_To_Date)
print(merge_final["YYYYWK_MONTH_CANONICAL"].head(100).to_string())
# %%
merge_final.drop(["DATE_INVOICE_GL_DATE", "RECEIPT_DATE", "YYYYWK_MONTH"], inplace=True, axis=1)
merge_final.rename(columns={"DATE_INVOICE_GL_DATE_CANONICAL":"DATE_INVOICE_GL_DATE", "RECEIPT_DATE_CANONICAL":"RECEIPT_DATE", "YYYYWK_MONTH_CANONICAL":"YYYYWK_MONTH"}, inplace=True)
# %%
print(f"merge_final.shape: {merge_final.shape}")
print(merge_final.head().to_string())
# %%
# CK added to convert col from str to int
# Snowflake Cell: cell222
merge_final['YYYYWK'] = np.floor(pd.to_numeric(merge_final['YYYYWK'], errors='coerce')).astype('Int64')
# %%
# Write merge_final to the disk instead of a Snowflake table
PS_DOCUWARE_RAW_DATA_EXTRACTION = outputs / "serialized_data" / "PS_DOCUWARE_RAW_DATA_EXTRACTION.joblib.gz"
# %%
if write_to_outputs:
    saved_path = JoblibSerializer().save_data(
        merge_final,
        PS_DOCUWARE_RAW_DATA_EXTRACTION
    )
# %%
# Read merge_final from a csv file to raw_df
# Fetching from previous SPROC output table
if read_from_outputs:
    raw_df = JoblibSerializer().load_data(PS_DOCUWARE_RAW_DATA_EXTRACTION)
    print(f"PS_DOCUWARE_RAW_DATA_EXTRACTION is loaded from {PS_DOCUWARE_RAW_DATA_EXTRACTION}")
    print(f"raw_df.shape: {raw_df.shape}")

print(raw_df.head().to_string())
print(f"raw_df.equals(merge_final): {raw_df.equals(merge_final)}")
# %%
# Here we try to save time
if not write_to_outputs and not read_from_outputs:
    raw_df = merge_final.copy(deep=True)
print(f"raw_df.equals(merge_final): {raw_df.equals(merge_final)}")
# We keep this extra copy because the variable can be redefined bleow.
raw_df_saved = merge_final.copy(deep=True)
# %% [markdown]
# # Data Imputation and Feature Engineering to make Training Set
# Snowflake Cell: cell156
non_ts_numeric_cols = ["PROBABILITY_OF_DELINQUENCY", "RICOH_CUSTOM_RISK_MODEL"]
non_ts_categorical_cols = ["OVERALL_BUSINESS_RISK", "PAYMENT_RISK_TRIPLE_A_RATING", "CONTRACT_LINE_ITEMS"]
columns_to_be_processed_later = non_ts_numeric_cols + non_ts_categorical_cols + ["CUST_ACCOUNT_NUMBER", "LIFESPAN_MONTHS", "DAYS_TO_CHURN"]
finalized_df_ohe_to_process = raw_df.groupby("CUST_ACCOUNT_NUMBER")[columns_to_be_processed_later].first()
print(finalized_df_ohe_to_process.head().to_string())
# %% Imputation for Non Time Series columns
pofd_median = finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"].median()
finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"] = finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"].apply(lambda x:   float(pofd_median) if np.isnan(x) else x)

temp=finalized_df_ohe_to_process["RICOH_CUSTOM_RISK_MODEL"].mode().to_frame()
rcrm_mode = temp.loc[temp.index[0], 'RICOH_CUSTOM_RISK_MODEL']
finalized_df_ohe_to_process["RICOH_CUSTOM_RISK_MODEL"] = finalized_df_ohe_to_process["RICOH_CUSTOM_RISK_MODEL"].apply(lambda x: float(rcrm_mode) if np.isnan(x) else x)

# One Hot Encoding for Categorical variables OVERALL_BUSINESS_RISK and PAYMENT_RISK_TRIPLE_A_RATING
finalized_df_ohe_to_process = finalized_df_ohe_to_process.reset_index(drop=True)

categorical_cols = ["OVERALL_BUSINESS_RISK", "PAYMENT_RISK_TRIPLE_A_RATING"]
fill_value = "UNK"

for col in categorical_cols:
    finalized_df_ohe_to_process[col].fillna(fill_value, inplace=True)
    finalized_df_ohe_to_process[col] = finalized_df_ohe_to_process[col].str.replace(" ", "_", regex=False)

    if col == "OVERALL_BUSINESS_RISK":
        col_abreviation = "obr_"
    else:
        col_abreviation = "prtar_"

    le_ohe = LabelEncoder()
    ohe = OneHotEncoder(handle_unknown = "ignore")
    enc_train = le_ohe.fit_transform(finalized_df_ohe_to_process[col]).reshape(finalized_df_ohe_to_process.shape[0],1)
    ohe_train = ohe.fit_transform(enc_train)
    le_ohe_name_mapping = dict(zip(le_ohe.classes_, le_ohe.transform(le_ohe.classes_)))

    enc_train = finalized_df_ohe_to_process[col].map(le_ohe_name_mapping).ravel().reshape(-1,1)
    enc_train[np.isnan(enc_train)] = 9999

    cols = [col_abreviation + str(x) for x in le_ohe_name_mapping.keys()]
    finalized_df_ohe_to_process = pd.concat([finalized_df_ohe_to_process.reset_index(), pd.DataFrame.sparse.from_spmatrix(ohe_train, columns = cols)], axis = 1).drop(["index"], axis=1)
    finalized_df_ohe_to_process.drop([col], axis = 1, inplace=True)

columns_to_be_droped = non_ts_categorical_cols+non_ts_numeric_cols
raw_df.drop(columns_to_be_droped, axis=1, inplace=True)

# Target Encoding for CONTRACT_LINE_ITEMS
finalized_df_ohe_to_process["CONTRACT_LINE_ITEMS"].fillna("NA", inplace=True)
finalized_df_ohe_to_process["CONTRACT_LINE_ITEMS"] = finalized_df_ohe_to_process["CONTRACT_LINE_ITEMS"].str.replace(r"\\d+x ", "", regex=True)

for i, row in finalized_df_ohe_to_process.iterrows():
    t = row["CONTRACT_LINE_ITEMS"]
    arr = t.split("-")
    arr = [x.strip() for x in arr]
    arr_s = sorted(arr)
    key = "-".join([s for s in arr_s])
    finalized_df_ohe_to_process.loc[i, "CONTRACT_LINE_ITEMS"] = key

X_train = finalized_df_ohe_to_process.copy()
y_train = finalized_df_ohe_to_process["LIFESPAN_MONTHS"]

enc = TargetEncoder(cols = ["CONTRACT_LINE_ITEMS"]).fit(X_train, y_train)
X_train_encoded = enc.transform(X_train)
print(X_train_encoded.head().to_string())
# %%
# Save the trained target encoder
now = datetime.now()
date_string = now.strftime("%Y%m%d")
ENC_CURRENT = outputs / "serialized_encoders" / 'ENC_CURRENT.joblib.gz'
# %%
# We may skip this cell to save time
if write_to_outputs:
    JoblibSerializer().save_data(enc, ENC_CURRENT)
    print(f"TargetEncoder saved to {ENC_CURRENT}")

if read_from_outputs:
    enc_loaded = JoblibSerializer().load_data(ENC_CURRENT)
    print(enc.get_params() == enc_loaded.get_params())

# %%%
# Imputation for Time Series columns
ts_columns = ["CUST_ACCOUNT_NUMBER", "YYYYWK", "DOCUMENTS_OPENED", "USED_STORAGE__MB", "INVOICE_REVLINE_TOTAL", "ORIGINAL_AMOUNT_DUE", "FUNCTIONAL_AMOUNT"]
raw_df["transformed_YYYYWK"] = raw_df["MONTH"].apply(convert_date_to_yyyywk)
# Impute missing YYYYWK with equivalent MONTH
raw_df["YYYYWK"].fillna(raw_df["transformed_YYYYWK"], inplace=True)
raw_df.drop("transformed_YYYYWK", axis=1, inplace=True)
ts_df = raw_df[ts_columns]
ts_df = ts_df[ts_df['YYYYWK'].notna()]
ts_df['YYYYWK'] = ts_df['YYYYWK'].astype(int)
ts_df['CUST_ACCOUNT_NUMBER'] = ts_df['CUST_ACCOUNT_NUMBER'].astype(int)
ts_df.rename(columns={'USED_STORAGE__MB':'USED_STORAGE_MB'},inplace=True)
ts_df_sorted = ts_df.sort_values(['CUST_ACCOUNT_NUMBER','YYYYWK']).drop_duplicates()
only_features = ts_df_sorted.copy()
only_features = only_features.fillna(0)
print(raw_df.head().to_string())
# %%
print(only_features.head().to_string())
# %%
lifespan_df = raw_df[['CUST_ACCOUNT_NUMBER', 'DAYS_TO_CHURN']]
lifespan_df['CUST_ACCOUNT_NUMBER'] = lifespan_df['CUST_ACCOUNT_NUMBER'].astype(int)
lifespan_df.drop_duplicates(inplace=True)
print(f"lifespan_df.shape: {lifespan_df.shape}")
print(lifespan_df.head().to_string())
# %%
# Snowflake Cell: cell46
df_churn_cust = l1_cust_churned[["CUST_ACCOUNT_NUMBER", "CHURN_DATE"]]
df_churn_cust['CHURN_YYYYWK_DATE'] = df_churn_cust['CHURN_DATE'].apply(convert_date_to_yyyywk)
df_churn_cust['CUST_ACCOUNT_NUMBER'] = df_churn_cust['CUST_ACCOUNT_NUMBER'].astype(int)
print(f"only_features.shape: {only_features.shape} # (14614, 7)")
df_raw_churn = only_features.merge(df_churn_cust, on='CUST_ACCOUNT_NUMBER', how='inner')
print(f"df_raw_churn.shape: {df_raw_churn.shape} # (14614, 9)")
final_features = df_raw_churn[df_raw_churn['YYYYWK'] <= df_raw_churn['CHURN_YYYYWK_DATE']]
print(f"final_features.shape: {final_features.shape}")
print(final_features.head().to_string())
# %%
final_features_for_FE = final_features.drop(['CHURN_DATE', 'CHURN_YYYYWK_DATE'], axis=1)
#print(final_features_for_FE.shape) # (5551, 7)
final_features_for_FE = final_features_for_FE.merge(lifespan_df, on='CUST_ACCOUNT_NUMBER', how='inner')
#print(final_features_for_FE.shape) # (5551, 8)
print(f"final_features_for_FE.shape: {final_features_for_FE.shape}")
print(final_features_for_FE.head().to_string())
# %%
print(X_train_encoded['CONTRACT_LINE_ITEMS'])
# %% [markdown]
# # Store time series and Non time series data into DB before feature engineering from TSFRESH
for col in X_train_encoded.columns:
    if pd.api.types.is_sparse(X_train_encoded[col]):
        X_train_encoded[col] = X_train_encoded[col].sparse.to_dense()
print(f"X_train_encoded.shape: {X_train_encoded.shape}")
print(X_train_encoded.head().to_string())
# %%
# all_usage = cell177.to_pandas()
# Direct SQL approach - replace your SQL with this:
query = """
WITH fi(CUST_ACCOUNT_NUMBER, YYYYWK) AS(
    SELECT CUST_ACCOUNT_NUMBER, min(YYYYWK) AS YY FROM RUS_AIML.DOCUWARE_USAGE_JAPAN_V1_LATEST_V GROUP BY(CUST_ACCOUNT_NUMBER)
)

SELECT a.CUST_ACCOUNT_NUMBER, a.YYYYWK FROM fi a join RUS_AIML.PS_DOCUWARE_L1_CUST b on a.CUST_ACCOUNT_NUMBER = b.CUST_ACCOUNT_NUMBER where b.CHURNED_FLAG='Y';
"""
all_usage = session.sql(query).to_pandas()
# %%
# we skeep this to save time
if read_from_outputs:
    raw_df = JoblibSerializer().load_data(PS_DOCUWARE_RAW_DATA_EXTRACTION)
    print(f"PS_DOCUWARE_RAW_DATA_EXTRACTION is read from {PS_DOCUWARE_RAW_DATA_EXTRACTION}")

print(f"raw_df.shape: {raw_df.shape}")
print(raw_df.head().to_string())
print(f"raw_df.equals(raw_df_saved): {raw_df.equals(raw_df_saved)}")
print(f"raw_df.shape: {raw_df.shape}")
print(f"raw_df_saved.shape: {raw_df_saved.shape}")
# %%
# We use this copy to save time
if not read_from_outputs:
    raw_df = raw_df_saved.copy(deep=True)
print(f"raw_df.shape: {raw_df.shape}")

print(raw_df.head().to_string())
# %%
df_cust_earliest_date = raw_df[['CUST_ACCOUNT_NUMBER', 'FINAL_EARLIEST_DATE']]
df_cust_earliest_date.drop_duplicates(inplace=True)
df_cust_earliest_date['CUST_ACCOUNT_NUMBER'] = df_cust_earliest_date['CUST_ACCOUNT_NUMBER'].astype(int)
all_usage['CUST_ACCOUNT_NUMBER'] = all_usage['CUST_ACCOUNT_NUMBER'].astype(int)
df_merged = pd.merge(df_cust_earliest_date, all_usage , on='CUST_ACCOUNT_NUMBER', how='inner')
print(f"df_merged.shape: {df_merged.shape}")
print(df_merged.head().to_string())
# %%
df_merged['CUST_ACCOUNT_NUMBER'] = df_merged['CUST_ACCOUNT_NUMBER'].astype(int)
final_features_for_FE['CUST_ACCOUNT_NUMBER'] = final_features_for_FE['CUST_ACCOUNT_NUMBER'].astype(int)
final_features_for_FE_trimmed = pd.DataFrame()

for ind,row in df_merged.iterrows():
    t = final_features_for_FE[ final_features_for_FE['CUST_ACCOUNT_NUMBER'] == row['CUST_ACCOUNT_NUMBER'] ]
    t['YYYYWK'] = t['YYYYWK'].astype(int)
    yyyywk_date = convert_yyyywk_to_actual_mid_date(row['YYYYWK'])

    row['YYYYWK'] = int(row['YYYYWK'])
    tt = t[ t['YYYYWK'] >= row['YYYYWK'] ]
    tt['DAYS_TO_CHURN'] = tt['DAYS_TO_CHURN'] - (pd.to_datetime(yyyywk_date) - pd.to_datetime(row['FINAL_EARLIEST_DATE'])).days
    final_features_for_FE_trimmed = pd.concat([tt, final_features_for_FE_trimmed], axis=0, ignore_index=True)
# %%
print(f"final_features_for_FE_trimmed.shape: {final_features_for_FE_trimmed.shape}")
print(final_features_for_FE_trimmed.head().to_string())
print()
print(final_features_for_FE_trimmed.tail().to_string())
print("\n".join(list(final_features_for_FE_trimmed.columns)))
# %%
# Write encoded data to csv files
# Snowflake Cell: cell43
# We may skip this cell to save time
PS_DOCUWARE_RAW_X_TRAIN_ENCODED_DATA_BEFORE_TSFRESH = outputs / "serialized_data" / "PS_DOCUWARE_RAW_X_TRAIN_ENCODED_DATA_BEFORE_TSFRESH.joblib.gz"
if write_to_outputs:
    saved_path = JoblibSerializer().save_data(
        X_train_encoded,
        PS_DOCUWARE_RAW_X_TRAIN_ENCODED_DATA_BEFORE_TSFRESH
    )
    print(f"'X_train_encoded' is written to {PS_DOCUWARE_RAW_X_TRAIN_ENCODED_DATA_BEFORE_TSFRESH}")


PS_DOCUWARE_RAW_DATA_BEFORE_TSFRESH = outputs / "serialized_data" / "PS_DOCUWARE_RAW_DATA_BEFORE_TSFRESH.joblib.gz"
if write_to_outputs:
    saved_path = JoblibSerializer().save_data(
        final_features_for_FE_trimmed,
        PS_DOCUWARE_RAW_DATA_BEFORE_TSFRESH
    )
    print(f"'final_features_for_FE_trimmed' is written to {PS_DOCUWARE_RAW_DATA_BEFORE_TSFRESH}")
# %% [markdown]
# Starting TSFRESH API: Elapsed: hh:mm:ss 0:33:17.075724
# For review with Justin and save time: Read the tsfresh features below ...
# Justin review skip
# We may sklip this cell to save time
if write_to_outputs:
    with timer():
        ts_comprehensive_df, final_ts_age  = engineer_timeseries_cols_using_tsfresh(final_features_for_FE_trimmed)
# %%
# Path where to write to and read from tsfreash features
ts_comprehensive_df_joblib_tsfresh = outputs / "serialized_features" / "tsfresh" / "ts_comprehensive_df.joblib.gz"
final_ts_age_joblib_tsfresh        = outputs / "serialized_features" / "tsfresh" / "final_ts_age.joblib.gz"
# %%
# Justin review: Skip ...
# We may skip this to save time
if write_to_outputs:
    JoblibSerializer().save_data(
        ts_comprehensive_df,
        ts_comprehensive_df_joblib_tsfresh
    )

    saved_path = JoblibSerializer().save_data(
        final_ts_age,
        final_ts_age_joblib_tsfresh
    )
# %%
# Justin review: Read ...
# We read it if we did not use tsfresh to save time
if read_from_outputs:
    ts_comprehensive_df = JoblibSerializer().load_data(ts_comprehensive_df_joblib_tsfresh)
    final_ts_age = JoblibSerializer().load_data(final_ts_age_joblib_tsfresh)
# %%
# Now we check everything
for col in final_features_for_FE_trimmed.columns:
    if re.search("^CUST_ACCOUNT_NUMBER", col):
        print(col)
# %%
print(f"ts_comprehensive_df.shape: {ts_comprehensive_df.shape}")
print(ts_comprehensive_df.head().to_string())
# %%
print("\n".join(list(ts_comprehensive_df.columns[:100])))
# %%
print(final_ts_age.head().to_string())
# %%
ts_comprehensive_df = ts_comprehensive_df.reset_index(drop=True)
final_ts_age = final_ts_age.reset_index(drop=True)
final_ts_df = pd.concat([ts_comprehensive_df, final_ts_age], axis=1, ignore_index=True)
list(final_ts_df.columns)
# %%
all_columns = ts_comprehensive_df.columns.tolist() + final_ts_age.columns.tolist()
print(f"set(final_ts_df.columns) == set(all_columns: {set(final_ts_df.columns) == set(all_columns)}")
# %%
final_ts_df.columns = all_columns
print(f"final_ts_df.shape: {final_ts_df.shape}")
print(final_ts_df.head().to_string())
# %%
# Justin Review: Skip ...
PS_DOCUWARE_TSFRESH_FEATURES_WITH_AGE = outputs / "serialized_features" / "tsfresh" / "PS_DOCUWARE_TSFRESH_FEATURES_WITH_AGE.joblib.gz"
# %%
# We may skip this if we want to save time
if write_to_outputs:
    saved_path = JoblibSerializer().save_data(
        final_ts_df,
        PS_DOCUWARE_TSFRESH_FEATURES_WITH_AGE
    )
    print(f"PS_DOCUWARE_RAW_DATA_EXTRACTION is written to {PS_DOCUWARE_TSFRESH_FEATURES_WITH_AGE}")
# %%
# Snowflake Cell: cell38
ts_comprehensive_df = final_ts_df.copy(deep=True)
print(f"ts_comprehensive_df.shape: {ts_comprehensive_df.shape}")
print(ts_comprehensive_df.head().to_string())
# %%
# Check if it contains CUST_ACCOUNT_NUMBER related TSFRESH features
for col in ts_comprehensive_df.columns:
    if re.search("^CUST_ACCOUNT_NUMBER", col):
        print(col)
    if re.search("^DAYS", col):
        print(col)
# %%
# Check first two words of all columns generated by TSFRESH
prefix=set()
for col in ts_comprehensive_df.columns:
    prefix.add((col.split('_')[0], col.split('_')[1]))
# %%
print(prefix)
# %% [markdown]
# # Storing Training Data after Feature Engineering and before Training
# Snowflake Cell: cell36
ts_comprehensive_df["CUST_ACCOUNT_NUMBER"] = ts_comprehensive_df["CUST_ACCOUNT_NUMBER"].astype(int)
X_train_encoded["CUST_ACCOUNT_NUMBER"] = X_train_encoded["CUST_ACCOUNT_NUMBER"].astype(int)
X_train_encoded = X_train_encoded.reset_index(drop=True)
ts_comprehensive_df = ts_comprehensive_df.reset_index(drop=True)
comprehensive_imputed_df = pd.merge(ts_comprehensive_df, X_train_encoded, on="CUST_ACCOUNT_NUMBER", how="inner")
# %%
print(f"ts_comprehensive_df.shape: {ts_comprehensive_df.shape}")
print(f"X_train_encoded.shape: {X_train_encoded.shape}")
print(f"comprehensive_imputed_df.shape: {comprehensive_imputed_df.shape}")
# %%
for col in ts_comprehensive_df.columns:
    if re.search("^DAYS", col):
        print(col)
# %%
for col in comprehensive_imputed_df.columns:
    if re.search("^DAYS", col):
        print(col)
# %%
# Snowflake Cell: cell36
for col in comprehensive_imputed_df.columns:
    if pd.api.types.is_sparse(comprehensive_imputed_df[col]):
        comprehensive_imputed_df[col] = comprehensive_imputed_df[col].sparse.to_dense()
# %%
PS_DOCUWARE_IMPUTED_DATA = outputs / "serialized_data"/ "PS_DOCUWARE_IMPUTED_DATA.joblib.gz"
# %%
# we may skip this to save time
if write_to_outputs:
    with timer():
        _ = JoblibSerializer().save_data(
            comprehensive_imputed_df,
            PS_DOCUWARE_IMPUTED_DATA
        )
        print(f"Successfuly created table PS_DOCUWARE_IMPUTED_DATA: {PS_DOCUWARE_IMPUTED_DATA}")
# %%
# We save it for future use
comprehensive_imputed_df_saved = comprehensive_imputed_df.copy(deep=True)
# %%
print(f"ts_comprehensive_df.shape: {ts_comprehensive_df.shape}")
print(f"comprehensive_imputed_df.shape: {comprehensive_imputed_df.shape}")
# %%
for col in ts_comprehensive_df.columns:
    if re.search("^DAYS", col):
        print(col)
# %%
for col in X_train_encoded.columns:
    if re.search("^DAYS", col):
        print(col)
# %%
for col in comprehensive_imputed_df.columns:
    if re.search("^DAYS", col):
        print(col)
# %%
# Check first two words of all columns generated by TSFRESH
prefix=set()

for col in comprehensive_imputed_df.columns:
    prefix.add((col.split('_')[0], col.split('_')[1]))
    prefix.add(col.split('_')[0])
# %%
print(prefix)
# %% [markdown]
# # Training using Catboost Regression
# We may skip this to save time
if read_from_outputs:
    with timer():
        comprehensive_imputed_df = JoblibSerializer().load_data(PS_DOCUWARE_IMPUTED_DATA)
        print(f"Successfuly read PS_DOCUWARE_IMPUTED_DATA: {PS_DOCUWARE_IMPUTED_DATA}")
# %%
# Check the corruption of the data after writing ot the disk
comprehensive_imputed_df.equals(comprehensive_imputed_df_saved)

comprehensive_imputed_df = comprehensive_imputed_df.replace([np.inf, -np.inf], 0)
comprehensive_imputed_df = comprehensive_imputed_df.fillna(0)
comprehensive_imputed_df.equals(comprehensive_imputed_df_saved)

with timer():
    for col in comprehensive_imputed_df.columns:
        comprehensive_imputed_df[col] = pd.to_numeric(comprehensive_imputed_df[col], errors="coerce").astype(float)

comprehensive_imputed_df.equals(comprehensive_imputed_df_saved)
# %% Experiment:
comprehensive_imputed_df = comprehensive_imputed_df_saved.copy(deep=True)
# comprehensive_imputed_o_df = pd.series(comprehensive_imputed_df)
# comprehensive_imputed_df = pd.to_numeric(comprehensive_imputed_o_df, errors="coerce")
# comprehensive_imputed_df = comprehensive_imputed_df.to_frame()
# %%
cust_churn_df = comprehensive_imputed_df[["CUST_ACCOUNT_NUMBER","DAYS_TO_CHURN"]]
print(cust_churn_df.head().to_string())
# %%
all_columns = comprehensive_imputed_df.columns
for cols in all_columns:
    if re.search("^CUST_ACCOUNT_NUMBER" , cols):
        print(cols)
# %%
cust_churn_df.drop_duplicates(inplace=True)
# %%
print(f"cust_churn_df.shape: {cust_churn_df.shape}")
print(cust_churn_df.head().to_string())
# %% [markdown]
# #### Stratified Sampling
# Snowflake Cell: cell102
# We have replaced the code here below
stratified_df = pd.DataFrame()

label_count=1
for i in range(365, 1461, 365):
    XX = cust_churn_df[((i-365) < cust_churn_df["DAYS_TO_CHURN"]) & (cust_churn_df["DAYS_TO_CHURN"] <= i)]
    XX["SAMPLE"] = label_count
    stratified_df = pd.concat([stratified_df, XX], axis=0, ignore_index=True)
    label_count+=1
    #print("For ", (i-365)+1, " To ", i , " Total Counts = ", XX.shape[0])
    if i==1460:
        XX = cust_churn_df[1460 < cust_churn_df["DAYS_TO_CHURN"]]
        XX["SAMPLE"] = label_count
        stratified_df = pd.concat([stratified_df, XX], axis=0, ignore_index=True)
        #print("For ", 1460+1, " To upper limit Total Counts = ", (1460 < data["DAYS_TO_CHURN"]).sum())
# %%
stratified_df["NO_OF_RENEWALS"] = stratified_df["SAMPLE"]-1
# %%
stratified_df["SAMPLE"].value_counts()
# %%
stratified_df["NO_OF_RENEWALS"].value_counts()
# %%
print(stratified_df.head().to_string())
# %%
stratified_df = stratified_df[stratified_df['SAMPLE']!=5]
features = stratified_df.drop(["DAYS_TO_CHURN", "SAMPLE"], axis=1)
y = stratified_df["SAMPLE"]
# %%
print(features)
# %%
print(y)
y.value_counts
# %%
# Comment is a bad solution ...
# X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
# X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, shuffle=True)
# %%
# # Imporved solution
# Initialize the sampler
sampler = StratifiedChurnSampler(
    min_samples_per_class=2,
    bin_days=[365, 730, 1095, 1460],
    random_state=42
)
# %%
# Run with all visualizations
results = sampler.fit_transform(
    cust_churn_df,
    test_size=0.2,
    keep_all_data=True,
    remove_last_bin=True,
    plot_distributions=True,     # Basic distributions
    plot_dashboard=True,         # 9-plot dashboard
    plot_verification=True       # Verification plots
)
# %%
# Get results
X_train = results['X_train']
X_test = results['X_test']
y_train = results['y_train']
y_test = results['y_test']
# %%
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# %%
y_train.value_counts()
# %%
y_test.value_counts()
# %%
X_test.head()
# %%
type(X_train), type(X_test), type(y_train), type(y_test)
# %%
# Additional analysis
sampler.print_summary()
sampler.plot_statistical_summary()
# %%
print(f"comprehensive_imputed_df.shape: {comprehensive_imputed_df.shape}")
print(comprehensive_imputed_df.head().to_string())
# %%
X_train_all_cols = pd.merge(comprehensive_imputed_df, X_train, how='inner', on='CUST_ACCOUNT_NUMBER')
print(f"X_train_all_cols.shape: {X_train_all_cols.shape}")
# %%
X_test_all_cols = pd.merge(comprehensive_imputed_df, X_test, how='inner', on='CUST_ACCOUNT_NUMBER')
print(f"X_test_all_cols.shape: {X_test_all_cols.shape}")
# %%
train_X = X_train_all_cols.drop(["CUST_ACCOUNT_NUMBER","LIFESPAN_MONTHS","DAYS_TO_CHURN", "DAYS_REMAINING"], axis=1) # Let cluster be in training
train_y = X_train_all_cols["DAYS_REMAINING"]
# %%
test_X = X_test_all_cols.drop(["CUST_ACCOUNT_NUMBER","LIFESPAN_MONTHS","DAYS_TO_CHURN", "DAYS_REMAINING"], axis=1) # Let cluster be in training
test_y = X_test_all_cols["DAYS_REMAINING"]
# %%
train_X.shape, train_y.shape, test_X.shape, test_y.shape
# %%
type(train_X), type(test_X), type(train_y), type(test_y)
# %%
import numpy as np
np.int = int
#above code to resolve
"""AttributeError: module 'numpy' has no attribute 'int'. `np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information. The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations"""
# %%
# #try:
# #Fetching from previous SPROC output table
# comprehensive_imputed_df = session.sql("SELECT * FROM RUS_AIML.PS_DOCUWARE_IMPUTED_DATA").to_pandas()
# comprehensive_imputed_df = comprehensive_imputed_df.replace([np.inf, -np.inf], 0)
# comprehensive_imputed_df = comprehensive_imputed_df.fillna(0)

# for col in comprehensive_imputed_df.columns:
#     comprehensive_imputed_df[col] = pd.to_numeric(comprehensive_imputed_df[col], errors="coerce").astype(float)

# # comprehensive_imputed_o_df = pd.series(comprehensive_imputed_df)
# # comprehensive_imputed_df = pd.to_numeric(comprehensive_imputed_o_df, errors="coerce")
# # comprehensive_imputed_df = comprehensive_imputed_df.to_frame()

# # Stratified Sampling
# stratified_df = pd.DataFrame()

# comprehensive_imputed_df[["CUST_ACCOUNT_NUMBER", "DAYS_TO_CHURN"]]

# label_count=1
# for i in range(365, 1461, 365):
#     XX = comprehensive_imputed_df[((i-365) < comprehensive_imputed_df["DAYS_TO_CHURN"]) & (comprehensive_imputed_df["DAYS_TO_CHURN"] <= i)]
#     #XX["SAMPLE"] = label_count
#     stratified_df = pd.concat([stratified_df, XX], axis=0, ignore_index=True)
#     #label_count+=1
#     #print("For ", (i-365)+1, " To ", i , " Total Counts = ", XX.shape[0])
#     if i==1460:
#         XX = comprehensive_imputed_df[1460 < comprehensive_imputed_df["DAYS_TO_CHURN"]]
#         XX["SAMPLE"] = label_count
#         stratified_df = pd.concat([stratified_df, XX], axis=0, ignore_index=True)
#         #print("For ", 1460+1, " To upper limit Total Counts = ", (1460 < data["DAYS_TO_CHURN"]).sum())


#stratified_df["LOG_OF_MONTHS_TO_CHURN"] = np.log(stratified_df["LIFESPAN_MONTHS"])

# Stratified Sampling
#features = stratified_df.drop(["CUST_ACCOUNT_NUMBER","LIFESPAN_MONTHS","DAYS_TO_CHURN", "SAMPLE"], axis=1)
# features = comprehensive_imputed_df.drop(["CUST_ACCOUNT_NUMBER","LIFESPAN_MONTHS","DAYS_TO_CHURN", "DAYS_REMAINING"], axis=1) # Let cluster be in training
# y = comprehensive_imputed_df["DAYS_REMAINING"]

# 20/80 Split
# X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, shuffle=True)#, stratify=y)
#y_train = X_train["LOG_OF_MONTHS_TO_CHURN"]
#y_test = X_test["LOG_OF_MONTHS_TO_CHURN"]

#X_train.drop("LOG_OF_MONTHS_TO_CHURN", axis=1, inplace=True)
#X_test.drop("LOG_OF_MONTHS_TO_CHURN", axis=1, inplace=True)

# Training

# param_grid = {"learning_rate": Real(0.2, .4, "uniform"), #was (.10,1)
#              "max_depth": Integer(9, 12),  #(2,12)
#              "subsample": Real(0.1, .2, "uniform"), # was (0.1, 1.0,
#              "colsample_bytree": Real(0.5, 1.0, "uniform"), # subsample ratio of columns by tree min was .1
#              "reg_lambda": Real(20., 40., "uniform"), # L2 regularization # was (1e-9, 100., "uniform")
#              "reg_alpha": Real(5., 20., "uniform"), # L1 regularization # was (1e-9, 100., "uniform")
#              "n_estimators": Integer(500, 1000) #was (20,3000)
# }

# # xgb_model = xgb.XGBRegressor(tree_method="gpu_hist", random_state=10)
# xgb_model = xgb.XGBRegressor(tree_method="auto", random_state=10)

# bayes_search = BayesSearchCV(
#         estimator=xgb_model,
#         search_spaces=param_grid,
#         scoring="neg_mean_squared_error",
#         cv=3,
#         n_iter=20,
#         n_jobs=-1,
#         verbose=3,
#         random_state=0)

# print("Training Starts")
# %%
param_grid = {
    "learning_rate": Real(0.01, 0.3, "log-uniform"),
    "depth": Integer(4, 8),                         # Tree depth
    "iterations": Integer(100, 800),                # Number of trees
    "subsample": Real(0.8, 1.0, "uniform"),        # Row sampling
    "l2_leaf_reg": Real(1e-6, 5., "log-uniform"),  # L2 regularization
}
# %%
catboost_model = cb.CatBoostRegressor(
    task_type='GPU',
    devices='0:3',              # Use all 4 GPUs
    bootstrap_type='Bernoulli', # Fixed type (no bagging_temperature)
    random_seed=10,
    verbose=100,
    allow_writing_files=False,
    gpu_ram_part=0.8,
)
# %%
bayes_search = BayesSearchCV(
    estimator=catboost_model,
    search_spaces=param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    n_iter=20,
    n_jobs=1, # critical -- check it
    verbose=3,
    random_state=0
)
print("Training Starts with CatBoost on 4 GPUs")
# %%
# Justin Review: Skip ... CPU Time ~ 21 min.
# We may skip trainng here and read the trained model
if write_to_outputs:
    print("Training Starts: Total time ~ 21 min.")
    with timer():
        bayes_search.fit(train_X, train_y)
        print("The model successfuly trained")
# %%
# bayes_search.fit(train_X, train_y)
# print("Training Finishes")
best_model = bayes_search.best_estimator_
best_model_path = outputs / "serialized_models"/"churn_model_v1.joblib.gz"
if write_to_outputs:
    saved_path = JoblibModelSerializer.save_model(
        model=best_model,
        save_path= best_model_path
    )
    print(f"The model successfuly stored in: {saved_path}" )

if read_from_outputs:
    best_model = JoblibModelSerializer.load_model(saved_path)
    print(f"The model successfuly restored from: {saved_path}" )

# # Make predictions
# predictions = model.predict(X_new)

#y_pred = best_models.predict(X_test)
#rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# import time
# import datetime
# ts = time.time()
# time_of_day = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d::%H:%M:%S')

# print("Writing model into Staging Archive")
# #import_dir = sys._xoptions.get("snowflake_import_directory")
# model_file = os.path.join("/tmp", "xgb.joblib.gz")
# dump(best_models, model_file)
# session.file.put(model_file, "@PS_DOCUWARE_CHURN/ps_docuware_churn_model",overwrite=True)

# print("Writing model into Staging")
# model_file = os.path.join("/tmp", "xgb."+str(time_of_day)+".joblib.gz")
# dump(best_models, model_file)
# session.file.put(model_file, "@PS_DOCUWARE_CHURN/ps_docuware_churn_model/archive",overwrite=True)

# print(f"Successfuly trained the model and stored: {saved_path}" )
#except Exception as e:
#print("FAIL(Post Processing)!" + " Error: " + str(e))
# %%

# %%
"""# Training
from snowflake.ml.modeling.distributors.xgboost import XGBEstimator, XGBScalingConfig

param_grid = {"learning_rate": Real(0.01, 1.0, "uniform"),
             "max_depth": Integer(2, 12),
             "subsample": Real(0.1, 1.0, "uniform"),
             "colsample_bytree": Real(0.1, 1.0, "uniform"), # subsample ratio of columns by tree
             "reg_lambda": Real(1e-9, 100., "uniform"), # L2 regularization
             "reg_alpha": Real(1e-9, 100., "uniform"), # L1 regularization
             "n_estimators": Integer(20, 3000)
}


scaling_config = XGBScalingConfig(use_gpu=True)

xgb_model = XGBEstimator( scaling_config = scaling_config)
#xgb_model = xgb.XGBRegressor(tree_method="gpu_hist", random_state=10)

bayes_search = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_iter=20,
        n_jobs=-1,
        verbose=3,
        #random_state=0
)

print("Training Starts")

bayes_search.fit(train_X, train_y)

print("Training Finishes")

best_models = bayes_search.best_estimator_

#y_pred = best_models.predict(X_test)
#rmse = np.sqrt(mean_squared_error(y_test, y_pred))

import time
import datetime
ts = time.time()
time_of_day = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d::%H:%M:%S')

print("Writing model into Staging Archive")
#import_dir = sys._xoptions.get("snowflake_import_directory")
model_file = os.path.join("/tmp", "xgb.joblib.gz")
dump(best_models, model_file)
session.file.put(model_file, "@PS_DOCUWARE_CHURN/ps_docuware_churn_model",overwrite=True)

print("Writing model into Staging")
model_file = os.path.join("/tmp", "xgb."+str(time_of_day)+".joblib.gz")
dump(best_models, model_file)
session.file.put(model_file, "@PS_DOCUWARE_CHURN/ps_docuware_churn_model/archive",overwrite=True)

print("Successfuly trained the model and stored" )
#except Exception as e:
#print("FAIL(Post Processing)!" + " Error: " + str(e))"""
# %% [markdown]
# # ----> Test for churned customers
# %%
# comprehensive_imputed_df.loc[test_X.index,"CUST_ACCOUNT_NUMBER"].astype(str).unique()
# %%
X_test_all_cols[ X_test_all_cols["DAYS_REMAINING"] == 0]
# %%
last_rows_X_test = X_test_all_cols[ X_test_all_cols["DAYS_REMAINING"] == 0]
# %%
last_rows_X_test[["CUST_ACCOUNT_NUMBER", "DAYS_REMAINING"]].index
# %%
test_25_X = last_rows_X_test.drop(["CUST_ACCOUNT_NUMBER","LIFESPAN_MONTHS","DAYS_TO_CHURN", "DAYS_REMAINING"], axis=1) # Let cluster be in training
test_25_y = last_rows_X_test["DAYS_REMAINING"]
# %%
test_25_X["NO_OF_RENEWALS"]
# %%
test_25_X.shape, test_25_y.shape
# %%
test_25_y
# %%
# # Inferencing
y_pred = best_model.predict(test_25_X)
remaining_months = y_pred/30
print(remaining_months.shape)

actual_remaining = test_25_y/30
final_test_predicted_df = pd.DataFrame({
    "CUST_ACCOUNT_NUMBER": last_rows_X_test.loc[test_25_X.index,"CUST_ACCOUNT_NUMBER"].astype(str),
    "PREDICTED_MONTHS_REMAINING": remaining_months,
    "ACTUAL_MONTH_REMAINING": actual_remaining,
    "RESIDUAL": abs(remaining_months-actual_remaining)
})
print(final_test_predicted_df.head(50).to_string())
# %%
y_pred = best_model.predict(test_25_X)
remaining_months = y_pred/30
print(remaining_months.shape)

actual_remaining = test_25_y/30
final_test_predicted_df = pd.DataFrame({
    "CUST_ACCOUNT_NUMBER": last_rows_X_test.loc[test_25_X.index,"CUST_ACCOUNT_NUMBER"].astype(str),
    "PREDICTED_MONTHS_REMAINING": remaining_months,
    "ACTUAL_MONTH_REMAINING": actual_remaining,
    "RESIDUAL": abs(remaining_months-actual_remaining)
})

# %%
# Initialize visualizer
viz = PredictionVisualizer()

# Use the churn analysis method for your Test 25 scenario
viz.plot_churn_analysis(
    predictions_df=final_test_predicted_df,
    pred_col="PREDICTED_MONTHS_REMAINING",
    actual_col="ACTUAL_MONTH_REMAINING",
    expected_value=0.0,  # Testing at churn point
    time_unit="months",
    thresholds=[1, 2, 3],
    title="Test 25: Predictions at Churn Point (25-Week Imputed Data)"
)
# %%
# comprehensive visualization
viz.plot_all(
    predictions_df=final_test_predicted_df,
    title_prefix="Test 25 Churn Predictions"
)
# %%
# Print summary
viz.print_summary(final_test_predicted_df)
# %%
# Let's analyze your results more clearly
viz = PredictionVisualizer()

# Add this analysis after your predictions
print("\n=== DETAILED ANALYSIS FOR TEST 25 ===")
print(f"Number of customers at churn point: {len(final_test_predicted_df)}")
print(f"\nActual values (should all be 0):")
print(f"  - Min: {final_test_predicted_df['ACTUAL_MONTH_REMAINING'].min()}")
print(f"  - Max: {final_test_predicted_df['ACTUAL_MONTH_REMAINING'].max()}")
print(f"  - Mean: {final_test_predicted_df['ACTUAL_MONTH_REMAINING'].mean()}")

print(f"\nPredicted values:")
print(f"  - Min: {final_test_predicted_df['PREDICTED_MONTHS_REMAINING'].min():.2f}")
print(f"  - Max: {final_test_predicted_df['PREDICTED_MONTHS_REMAINING'].max():.2f}")
print(f"  - Mean: {final_test_predicted_df['PREDICTED_MONTHS_REMAINING'].mean():.2f}")
print(f"  - Median: {final_test_predicted_df['PREDICTED_MONTHS_REMAINING'].median():.2f}")

# Better metrics for churn point testing
errors = final_test_predicted_df['RESIDUAL']
print(f"\nTiming Error Analysis:")
print(f"  - Average months early/late: {final_test_predicted_df['PREDICTED_MONTHS_REMAINING'].mean():.2f}")
print(f"  - Customers predicted within 1 month: {(errors <= 1).sum()} ({(errors <= 1).mean()*100:.1f}%)")
print(f"  - Customers predicted within 3 months: {(errors <= 3).sum()} ({(errors <= 3).mean()*100:.1f}%)")
print(f"  - Customers predicted within 6 months: {(errors <= 6).sum()} ({(errors <= 6).mean()*100:.1f}%)")

# Show the distribution
plt.figure(figsize=(10, 5))
plt.hist(final_test_predicted_df['PREDICTED_MONTHS_REMAINING'], bins=20, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', lw=2, label='Expected (0 months)')
plt.axvline(x=final_test_predicted_df['PREDICTED_MONTHS_REMAINING'].mean(), color='green',
            linestyle='--', lw=2, label=f'Mean prediction ({final_test_predicted_df["PREDICTED_MONTHS_REMAINING"].mean():.2f} months)')
plt.xlabel('Predicted Months Remaining')
plt.ylabel('Count')
plt.title('Test 25: Model Predictions at Churn Point\n(All actual values = 0)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# %%
# Load Training data
# comprehensive_imputed_df = session.sql("SELECT * FROM RUS_AIML.PS_DOCUWARE_IMPUTED_DATA").to_pandas()
# comprehensive_imputed_df = comprehensive_imputed_df.replace([np.inf, -np.inf], 0)
# comprehensive_imputed_df = comprehensive_imputed_df.fillna(0)

# for col in comprehensive_imputed_df.columns:
#     comprehensive_imputed_df[col] = pd.to_numeric(comprehensive_imputed_df[col], errors="coerce").astype(float)

# Stratified Sampling
#features = stratified_df.drop(["CUST_ACCOUNT_NUMBER","LIFESPAN_MONTHS","DAYS_TO_CHURN", "SAMPLE"], axis=1)
# features = comprehensive_imputed_df.drop(["CUST_ACCOUNT_NUMBER","LIFESPAN_MONTHS","DAYS_TO_CHURN", "DAYS_REMAINING"], axis=1) # Let cluster be in training
# y = comprehensive_imputed_df["DAYS_REMAINING"]

# # 20/80 Split
# X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, shuffle=True)#, stratify=y)

# # Load model_file
# model_file = os.path.join('/tmp', 'xgb.joblib.gz')
# session.file.get("@PS_DOCUWARE_CHURN/ps_docuware_churn_model/xgb.joblib.gz", "/tmp")
# best_models = load(model_file)

# # Split XTrain,Ytrain
# Inferencing
y_pred = best_model.predict(test_X)
remaining_months = y_pred/30
print(remaining_months.shape)

actual_remaining = test_y/30
final_test_predicted_df = pd.DataFrame({
    "CUST_ACCOUNT_NUMBER": comprehensive_imputed_df.loc[test_X.index,"CUST_ACCOUNT_NUMBER"].astype(str),
    "PREDICTED_MONTHS_REMAINING": remaining_months,
    "ACTUAL_MONTH_REMAINING": actual_remaining,
    "RESIDUAL": abs(remaining_months-actual_remaining)
})
print(final_test_predicted_df.head(796).to_string())
# %%
# Use the new lifecycle analysis
viz = PredictionVisualizer()

viz.plot_lifecycle_analysis(
    predictions_df=final_test_predicted_df,
    pred_col="PREDICTED_MONTHS_REMAINING",
    actual_col="ACTUAL_MONTH_REMAINING",
    time_unit="months",
    cohort_size=3,  # Group by 3-month cohorts
    title="General Test Set - Lifecycle Prediction Analysis"
)

# %%
# Create monthly churn forecast
viz.plot_churn_forecast(
    predictions_df=final_test_predicted_df,
    pred_col="PREDICTED_MONTHS_REMAINING",
    base_date=pd.Timestamp.today(),  # Or any reference date
    n_months=12,  # Forecast next 12 months
    title="Monthly Churn Forecast Based on Model Predictions"
)
# %%
# %% This returns a dataframe with monthly predictions
forecast_df = viz.plot_churn_forecast(
    predictions_df=final_test_predicted_df,
    n_months=24  # 2-year forecast
)
# %%
# %%
# Feature Importance Scores
# importance = best_models.get_booster().get_score(importance_type='gain')
feature_importance = best_model.get_feature_importance()
feature_names = best_model.feature_names_

# Create importance dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("Top 20 Most Important Features:")
print(importance_df.head(20))

# Plot feature importance
plt.figure(figsize=(40,30))
top_features = importance_df.head(20)
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=36)
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importance - CatBoost Model')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
# %%
# importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Score'])
importance_df.head(287)
# %% [markdown]
# # Plot of Important Features ranked by XGBoost algorithm
# %%
importance_df
# # %%
# plt.figure(figsize=(40,30))
# xgb.plot_importance(best_models, max_num_features=20)
# #plt.savefig('C:/My Documents/work/Projects/Short Term Goal 2 - Customer Churn Prediction/ML Modeling Phase/Overall_Model_Importance_16Apr_2025.png')
# plt.show()

# %% [markdown]
# # Data extraction of Live Customers for Inferencing
# ------------------------------------------------------------------
# %%
# Code to impute missing usage data for first half of 2023
# as Impute_missing_Usage_23
# OLD CODE: usage_latest = session.sql("SELECT * FROM RUS_AIML.DOCUWARE_USAGE_JAPAN_V1_LATEST_V").to_pandas()
usage_latest = session.sql("""
    WITH latback AS (
        SELECT DISTINCT CONTRACT_NUMBER,
               REGEXP_REPLACE(trim(CUSTOMER_NAME), '  ', ' ') AS CUSTOMER_NAME,
               CONTRACT_START,
               CONTRACT_END,
               DOCUMENTS_OPENED,
               USED_STORAGE__MB,
               CONTRACT_LINE_ITEMS,
               PERIOD,
               YYYYWK
        FROM RUS_AIML.DOCUWARE_USAGE_COMBINED_V
    ),
    jaro AS (
        SELECT DISTINCT a.CUST_ACCOUNT_NUMBER,
               a.churn_date,
               a.churned_flag,
               b.CONTRACT_NUMBER,
               a.CUST_PARTY_NAME,
               b.CUSTOMER_NAME,
               b.CONTRACT_START,
               b.CONTRACT_END,
               b.DOCUMENTS_OPENED,
               b.USED_STORAGE__MB,
               b.CONTRACT_LINE_ITEMS,
               b.PERIOD,
               b.YYYYWK,
               jarowinkler_similarity(a.CUST_PARTY_NAME, b.CUSTOMER_NAME),
               rank() OVER (
                   PARTITION BY a.CUST_ACCOUNT_NUMBER
                   ORDER BY jarowinkler_similarity(a.CUST_PARTY_NAME, b.CUSTOMER_NAME) DESC
               ) AS match_rank
        FROM RUS_AIML.PS_DOCUWARE_CUST_FINAL a
        FULL OUTER JOIN latback b
        WHERE jarowinkler_similarity(a.CUST_PARTY_NAME, b.CUSTOMER_NAME) BETWEEN 93 AND 100
    )
    SELECT *
    FROM jaro
    WHERE jaro.match_rank = 1
""").to_pandas()

# %%
usage_latest['CONTRACT_START'] = pd.to_datetime(usage_latest['CONTRACT_START'])
usage_latest['CHURN_DATE']= pd.to_datetime(usage_latest['CHURN_DATE'])
usage_latest['YYYYWK'] = usage_latest['YYYYWK'].astype('Int64')

#Focus only on Contract Start Date >= ‘2020-01-01’
usageFIX = usage_latest[usage_latest['CONTRACT_START']>= '2020-01-01' ]

# Active -  If contract start date <= 2022-12-31  for active then we need to add weekly usage numbers ( average of last 10 for numerical)
usageFIXActive = usageFIX[(usageFIX['CONTRACT_START']<= '2022-12-31') & (usageFIX['CHURNED_FLAG']== 'N') & (usageFIX['YYYYWK'] <= 202252) ]

idx = usageFIXActive.groupby('CUST_ACCOUNT_NUMBER')['YYYYWK'].idxmax()
max_scores = usageFIXActive.loc[idx]
df_duplicated = pd.DataFrame(np.repeat(max_scores.values, repeats=25, axis=0), columns=max_scores.columns)
df_duplicated= df_duplicated.sort_values(by = 'CUST_ACCOUNT_NUMBER')

#range of YYYYWK missing data
recurring_range = []

for i in range(1,26):

    if i < 10:
        month_date = str(2023)+"0"+str(i)
    else:
        month_date = str(2023)+str(i)
    recurring_range.append(month_date)

# Calculate how many times the range needs to repeat
num_repeats = len(max_scores)    #len(df) // len(recurring_range) + 1

# Create the new column values by repeating the range and truncating
new_column = np.tile(recurring_range, num_repeats)[:len(df_duplicated)]

# Replace the column
df_duplicated['YYYYWK'] = new_column

# Active account imputations
ActiveAccts = df_duplicated

#End Active customer imputation

#Churned Customer imputation

# Churned Customers part 1    If contract start date <= 2022-12-31  and churned date after > 2022-12-31 then add in usage for wks between 202301-202325
usageFIXChurned1 = usageFIX[(usageFIX['CONTRACT_START']<= '2022-12-31') & (usageFIX['CHURNED_FLAG']== 'Y') &  (usageFIX['CHURN_DATE'] > '2023-06-24' ) & (usageFIX['YYYYWK'] <= 202252)   ]

idx = usageFIXChurned1.groupby('CUST_ACCOUNT_NUMBER')['YYYYWK'].idxmax()
max_scores = usageFIXChurned1.loc[idx]
df_duplicated = pd.DataFrame(np.repeat(max_scores.values, repeats=25, axis=0), columns=max_scores.columns)
df_duplicated.sort_values(by = 'CUST_ACCOUNT_NUMBER')

# Calculate how many times the range needs to repeat
num_repeats = len(max_scores)   #len(df) // len(recurring_range) + 1

# Create the new column values by repeating the range and truncating
new_column = np.tile(recurring_range, num_repeats)[:len(df_duplicated)]

# Replace the column
df_duplicated['YYYYWK'] = new_column

# save imputations for part 1 of churned customers
churned1 = df_duplicated

# end churned customer part1

# Churned Customers part 2 those customers who churned during the data outage
usageFIXChurned2 = usageFIX[(usageFIX['CONTRACT_START']<= '2022-12-31') & (usageFIX['CHURNED_FLAG']== 'Y') &  (usageFIX['CHURN_DATE'] <= '2023-06-24' ) & (usageFIX['YYYYWK'] <= 202252) & (usageFIX['CHURN_DATE'] >= '2023-01-01' )  ]

idx = usageFIXChurned2.groupby('CUST_ACCOUNT_NUMBER')['YYYYWK'].idxmax()
max_scores = usageFIXChurned2.loc[idx]
max_scores['ywk'] = [(i -pd.to_datetime('2023-01-01')).days//7 for i in max_scores['CHURN_DATE']]

# Method to duplicate rows
df_duplicated = max_scores.loc[np.repeat(max_scores.index.values, max_scores['ywk'])]

# Reset index if needed
df_duplicated = df_duplicated.reset_index(drop=True)

# Replace hard coded range
start_index = 0
end_index = 23
new_values = recurring_range[0:24]

df_duplicated.loc[start_index:end_index, 'YYYYWK'] = new_values

#  Replace hard coded range
start_index = 24
end_index = 46
new_values = recurring_range[0:23]

df_duplicated.loc[start_index:end_index, 'YYYYWK'] = new_values

#  Replace hard coded range
start_index = 47
end_index = 56
new_values = recurring_range[0:10]

df_duplicated.loc[start_index:end_index, 'YYYYWK'] = new_values

# Replace hard coded range
start_index = 57
end_index = 80
new_values = recurring_range[0:24]

df_duplicated.loc[start_index:end_index, 'YYYYWK'] = new_values

# Replace hard coded range
start_index = 81
end_index = 87
new_values = recurring_range[0:7]

df_duplicated.loc[start_index:end_index, 'YYYYWK'] = new_values

# save imputations for part 2 of churned customers
churned2 = df_duplicated

#Combine imputations
imputations = pd.concat([ActiveAccts,churned1,churned2])
# drop added columns from imputations
imputations = imputations.drop(columns=['ywk'])

usage = pd.concat([usageFIX,imputations])

usage_latest = usage
# %%
print(usage['YYYYWK'])
# %%
# %% PREPROCESSING RAW DATA
# Fetching all the different views
#  Sproc_2
#  Justin Bovard 2025-07-15
pay_df = session.sql("SELECT CUSTOMER_NO, RECEIPT_DATE, FUNCTIONAL_AMOUNT FROM RUS_AIML.PS_DOCUWARE_PAYMENTS").to_pandas()
rev_df = session.sql("SELECT CUST_ACCOUNT_NUMBER, DATE_INVOICE_GL_DATE, INVOICE_REVLINE_TOTAL FROM RUS_AIML.PS_DOCUWARE_REVENUE").to_pandas()
trx_df = session.sql("SELECT ACCOUNT_NUMBER, TRX_DATE, ORIGINAL_AMOUNT_DUE FROM RUS_AIML.PS_DOCUWARE_TRX").to_pandas()
contracts_sub_df = session.sql("SELECT CUST_ACCOUNT_NUMBER, SLINE_START_DATE, SLINE_END_DATE, SLINE_STATUS FROM RUS_AIML.PS_DOCUWARE_CONTRACT_SUBLINE").to_pandas()
renewals_df = session.sql("SELECT TO_CHAR(BILLTOCUSTOMERNUMBER) AS BILLTOCUSTOMERNUMBER, TO_CHAR(SHIPTOCUSTNUM) AS SHIPTOCUSTNUM, STARTDATECOVERAGE, CONTRACT_END_DATE FROM RUS_AIML.PS_DOCUWARE_SSCD_RENEWALS").to_pandas()
l1_cust_df =  session.sql("SELECT CUST_ACCOUNT_NUMBER, CUST_PARTY_NAME, L3_RISE_CONSOLIDATED_NUMBER, L3_RISE_CONSOLIDATED_NAME, L2_RISE_CONSOLIDATED_NUMBER, L2_RISE_CONSOLIDATED_NAME, CUST_ACCOUNT_TYPE, CUSTOMER_SEGMENT, CUSTOMER_SEGMENT_LEVEL, CHURNED_FLAG, CHURN_DATE FROM RUS_AIML.PS_DOCUWARE_L1_CUST").to_pandas()
dnb_risk_df = session.sql("SELECT TO_CHAR(ACCOUNT_NUMBER) AS ACCOUNT_NUMBER, OVERALL_BUSINESS_RISK, RICOH_CUSTOM_RISK_MODEL, PROBABILITY_OF_DELINQUENCY, PAYMENT_RISK_TRIPLE_A_RATING FROM RUS_AIML.DNB_RISK_BREAKDOWN").to_pandas()
usage_latest = session.sql("SELECT * FROM RUS_AIML.DOCUWARE_USAGE_JAPAN_V1_LATEST_V").to_pandas()

# Removing duplicates from all the pandas dataframes
pay_df = pay_df.drop_duplicates()
rev_df = rev_df.drop_duplicates()
trx_df = trx_df.drop_duplicates()
contracts_sub_df = contracts_sub_df.drop_duplicates()
renewals_df = renewals_df.drop_duplicates()
l1_cust_df = l1_cust_df.drop_duplicates()
dnb_risk_df = dnb_risk_df.drop_duplicates()
usage_latest = usage_latest.drop_duplicates()

# Selecting active customers

l1_cust_active = l1_cust_df[l1_cust_df["CHURNED_FLAG"]=='N']

pay_df_active = pay_df.merge(l1_cust_active[["CUST_ACCOUNT_NUMBER"]], left_on = "CUSTOMER_NO", right_on = "CUST_ACCOUNT_NUMBER", how="inner")
pay_df_active = pay_df_active.drop("CUSTOMER_NO", axis=1)
pay_df_active["MONTH"] = pd.to_datetime(pay_df_active["RECEIPT_DATE"]).dt.to_period("M").dt.to_timestamp()

rev_df_active = rev_df.merge(l1_cust_active[["CUST_ACCOUNT_NUMBER"]], on = "CUST_ACCOUNT_NUMBER", how="inner")
rev_df_active["MONTH"] = pd.to_datetime(rev_df_active["DATE_INVOICE_GL_DATE"]).dt.to_period("M").dt.to_timestamp()

trx_df_active = trx_df.merge(l1_cust_active[["CUST_ACCOUNT_NUMBER"]], left_on = "ACCOUNT_NUMBER", right_on = "CUST_ACCOUNT_NUMBER", how="inner")
trx_df_active = trx_df_active.drop("ACCOUNT_NUMBER", axis=1)
trx_df_active["MONTH"] = pd.to_datetime(trx_df_active["TRX_DATE"]).dt.to_period("M").dt.to_timestamp()

p_r_t_merged = pay_df_active.merge(rev_df_active, on = ["CUST_ACCOUNT_NUMBER","MONTH"], how="outer").merge(trx_df_active, on = ["CUST_ACCOUNT_NUMBER","MONTH"], how="outer")

#Contracts Subline
contracts_sub_df["SLINE_START_DATE"] = pd.to_datetime(contracts_sub_df["SLINE_START_DATE"])
contracts_sub_df["SLINE_END_DATE"] = pd.to_datetime(contracts_sub_df["SLINE_END_DATE"])
contracts_sub_df["SUB_START_MONTH"] = pd.to_datetime(contracts_sub_df["SLINE_START_DATE"]).dt.to_period("M").dt.to_timestamp()
contracts_sub_df["SUB_END_MONTH"] = pd.to_datetime(contracts_sub_df["SLINE_END_DATE"]).dt.to_period("M").dt.to_timestamp()
contracts_sub_df["SUB_EARLIEST_MONTH"] = contracts_sub_df.groupby("CUST_ACCOUNT_NUMBER")["SUB_START_MONTH"].transform("min")
contracts_sub_df["SUB_LATEST_MONTH"] = contracts_sub_df.groupby("CUST_ACCOUNT_NUMBER")["SUB_END_MONTH"].transform("max")
contracts_sub_df_active = contracts_sub_df.merge(l1_cust_active[["CUST_ACCOUNT_NUMBER"]], on = "CUST_ACCOUNT_NUMBER", how="inner")

contracts_sub_df_active = contracts_sub_df_active.drop_duplicates()

# Renewals
cols_to_str = ["BILLTOCUSTOMERNUMBER","SHIPTOCUSTNUM"]
renewals_df[cols_to_str] = renewals_df[cols_to_str].astype("Int64").astype(str)

renewals_df_active_1 = renewals_df.merge(l1_cust_active[["CUST_ACCOUNT_NUMBER"]], left_on = "SHIPTOCUSTNUM", right_on = "CUST_ACCOUNT_NUMBER",  how="left")

renewals_df_active_2 = renewals_df_active_1.merge(l1_cust_active[["CUST_ACCOUNT_NUMBER"]], left_on = "BILLTOCUSTOMERNUMBER", right_on = "CUST_ACCOUNT_NUMBER",  how="left", suffixes = ("_1", "_2"))
renewals_df_active_2["CUST_ACCOUNT_NUMBER"] = renewals_df_active_2["CUST_ACCOUNT_NUMBER_1"].fillna(renewals_df_active_2["CUST_ACCOUNT_NUMBER_2"])
renewals_df_active_2 = renewals_df_active_2.drop(["BILLTOCUSTOMERNUMBER", "SHIPTOCUSTNUM","CUST_ACCOUNT_NUMBER_1","CUST_ACCOUNT_NUMBER_2"], axis=1)
renewals_df_active_2 = renewals_df_active_2.dropna()

renewals_df_active_2["STARTDATECOVERAGE"] =  pd.to_datetime(renewals_df_active_2["STARTDATECOVERAGE"])
renewals_df_active_2["RENEWALS_START_MONTH"] = pd.to_datetime(renewals_df_active_2["STARTDATECOVERAGE"]).dt.to_period("M").dt.to_timestamp()
renewals_df_active_2["RENEWALS_END_MONTH"] = pd.to_datetime(renewals_df_active_2["CONTRACT_END_DATE"]).dt.to_period("M").dt.to_timestamp()
renewals_df_active_2["RENEWALS_EARLIEST_MONTH"] = renewals_df_active_2.groupby(renewals_df_active_2["CUST_ACCOUNT_NUMBER"])["RENEWALS_START_MONTH"].transform("min")
renewals_df_active_2["RENEWALS_LATEST_MONTH"] = renewals_df_active_2.groupby(renewals_df_active_2["CUST_ACCOUNT_NUMBER"])["RENEWALS_END_MONTH"].transform("max")

#DNB Risk Breakdown
dnb_risk_df["ACCOUNT_NUMBER"] = dnb_risk_df["ACCOUNT_NUMBER"].astype("Int64").astype(str)
dnb_risk_df_active = dnb_risk_df.merge(l1_cust_active[["CUST_ACCOUNT_NUMBER"]], left_on = "ACCOUNT_NUMBER", right_on = "CUST_ACCOUNT_NUMBER",  how="inner")
dnb_risk_df_active = dnb_risk_df_active.drop("ACCOUNT_NUMBER", axis=1)

# %% TESTIT
usage_latest["YYYYWK_Transformed"] = pd.to_datetime(usage_latest["YYYYWK"].apply(convert_yyyywk_to_date), errors = "coerce")

#Usage Japan Latest
usage_latest["CUST_ACCOUNT_NUMBER"] = usage_latest["CUST_ACCOUNT_NUMBER"].astype("Int64").astype(str)

usage_latest["YYYYWK_MONTH"] = pd.to_datetime(usage_latest["YYYYWK_Transformed"]).dt.to_period("M").dt.to_timestamp()

usage_latest_active = usage_latest.merge(l1_cust_active[["CUST_ACCOUNT_NUMBER"]], on="CUST_ACCOUNT_NUMBER",  how="inner")

# Merging all the active customers data frames i.e. Payments, Revenue, Transactions, contracts, contracts subline, contracts topline, renewals, snow inc, tech survey, loyalty survey, dnb risk and usage latest

merged_1 = p_r_t_merged.merge(contracts_sub_df_active, left_on = ["CUST_ACCOUNT_NUMBER", "MONTH"], right_on = ["CUST_ACCOUNT_NUMBER", "SUB_START_MONTH"], how="outer")

merged_1["MONTH"] = merged_1["MONTH"].fillna(merged_1["SUB_START_MONTH"])
merged_1 = merged_1.drop("SUB_START_MONTH", axis=1)
merged_2 = merged_1.merge(renewals_df_active_2, left_on = ["CUST_ACCOUNT_NUMBER","MONTH"], right_on = ["CUST_ACCOUNT_NUMBER","RENEWALS_START_MONTH"], how="outer")
merged_2["MONTH"] = merged_2["MONTH"].fillna(merged_2["RENEWALS_START_MONTH"])
merged_2 = merged_2.drop("RENEWALS_START_MONTH", axis=1)

merged_3 = merged_2.merge(dnb_risk_df_active, on="CUST_ACCOUNT_NUMBER", how="left")

merged_4 = merged_3.merge(usage_latest_active, left_on= ["CUST_ACCOUNT_NUMBER","MONTH"], right_on = ["CUST_ACCOUNT_NUMBER", "YYYYWK_MONTH"], how="outer")
merged_4["MONTH"] = merged_4["MONTH"].fillna(merged_4["YYYYWK_MONTH"])

to_drop_2 = ["CONTRACT_NUMBER", "CUST_PARTY_NAME", "CUSTOMER_NAME", "CONTRACT_END","JAROWINKLER_SIMILARITY(A.CUST_PARTY_NAME, B.CUSTOMER_NAME)"]
#(A.CUST_PARTY_NAME, B.CUSTOMER_NAME)

merged_5 = merged_4.drop(to_drop_2, axis=1)

merged_5 = merged_4.merge(l1_cust_active[["CUST_ACCOUNT_NUMBER", "CHURNED_FLAG"]], on = "CUST_ACCOUNT_NUMBER", how="inner")
merged_5["EARLIEST_DATE"] = merged_5[["RECEIPT_DATE", "DATE_INVOICE_GL_DATE", "TRX_DATE", "SLINE_START_DATE", "STARTDATECOVERAGE"]].min(axis=1)
merged_5["FINAL_EARLIEST_DATE"] = merged_5.groupby("CUST_ACCOUNT_NUMBER")["EARLIEST_DATE"].transform("min")

to_drop_3 = ["RECEIPT_DATE", "DATE_INVOICE_GL_DATE", "TRX_DATE", "SLINE_START_DATE", "SLINE_END_DATE", "SUB_END_MONTH","SLINE_STATUS", "SUB_EARLIEST_MONTH", "SUB_LATEST_MONTH", "STARTDATECOVERAGE", "CONTRACT_END_DATE", "RENEWALS_END_MONTH", "RENEWALS_EARLIEST_MONTH","RENEWALS_LATEST_MONTH", "YYYYWK_MONTH", "PERIOD", "CHURNED_FLAG", "EARLIEST_DATE"]

merged_5 = merged_5.drop(to_drop_3, axis=1)

date_col = ["MONTH","YYYYWK_Transformed", "FINAL_EARLIEST_DATE"]

merged_5[date_col] = merged_5[date_col].astype(str)

merged_5 = merged_5.drop_duplicates()

#CK added to convert col from str to int
merged_5['YYYYWK'] = np.floor(pd.to_numeric(merged_5['YYYYWK'], errors='coerce')).astype('Int64')

# session.write_pandas(merged_5, "PS_DOCUWARE_RAW_DATA_PREDICTION", auto_create_table=True, overwrite = True)
# %% [markdown]
# # Data Preprocessing of Live customers
# Fetching from previous SPROC output table
raw_df_active = session.sql("SELECT * FROM RUS_AIML.PS_DOCUWARE_RAW_DATA_PREDICTION").to_pandas()

non_ts_numeric_cols = ["PROBABILITY_OF_DELINQUENCY", "RICOH_CUSTOM_RISK_MODEL"]
non_ts_categorical_cols = ["OVERALL_BUSINESS_RISK", "PAYMENT_RISK_TRIPLE_A_RATING", "CONTRACT_LINE_ITEMS"]
columns_to_be_processed_later = non_ts_numeric_cols + non_ts_categorical_cols + ["CUST_ACCOUNT_NUMBER"]
finalized_df_ohe_to_process = raw_df_active.groupby("CUST_ACCOUNT_NUMBER")[columns_to_be_processed_later].first()


# %%
# Imputation for Non Time Series columns
pofd_median = finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"].median()
finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"] = finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"].apply(lambda x:   float(pofd_median) if np.isnan(x) else x)

rcrm_mode = finalized_df_ohe_to_process["RICOH_CUSTOM_RISK_MODEL"].mode()
finalized_df_ohe_to_process["RICOH_CUSTOM_RISK_MODEL"] = finalized_df_ohe_to_process["RICOH_CUSTOM_RISK_MODEL"].apply(lambda x: float(rcrm_mode) if np.isnan(x) else x)

# %%
# One Hot Encoding for Categorical variables OVERALL_BUSINESS_RISK and PAYMENT_RISK_TRIPLE_A_RATING
finalized_df_ohe_to_process = finalized_df_ohe_to_process.reset_index(drop=True)

categorical_cols = ["OVERALL_BUSINESS_RISK", "PAYMENT_RISK_TRIPLE_A_RATING"]
fill_value = "UNK"

for col in categorical_cols:
    finalized_df_ohe_to_process[col].fillna(fill_value, inplace=True)
    finalized_df_ohe_to_process[col] = finalized_df_ohe_to_process[col].str.replace(" ", "_", regex=False)

    if col == "OVERALL_BUSINESS_RISK":
        col_abreviation = "obr_"
    else:
        col_abreviation = "prtar_"

    le_ohe = LabelEncoder()
    ohe = OneHotEncoder(handle_unknown = "ignore")
    enc_train = le_ohe.fit_transform(finalized_df_ohe_to_process[col]).reshape(finalized_df_ohe_to_process.shape[0],1)
    ohe_train = ohe.fit_transform(enc_train)
    le_ohe_name_mapping = dict(zip(le_ohe.classes_, le_ohe.transform(le_ohe.classes_)))

    enc_train = finalized_df_ohe_to_process[col].map(le_ohe_name_mapping).ravel().reshape(-1,1)
    enc_train[np.isnan(enc_train)] = 9999

    cols = [col_abreviation + str(x) for x in le_ohe_name_mapping.keys()]
    finalized_df_ohe_to_process = pd.concat([finalized_df_ohe_to_process.reset_index(), pd.DataFrame.sparse.from_spmatrix(ohe_train, columns = cols)], axis = 1).drop(["index"], axis=1)
    finalized_df_ohe_to_process.drop([col], axis = 1, inplace=True)

columns_to_be_droped = non_ts_categorical_cols+non_ts_numeric_cols
raw_df_active.drop(columns_to_be_droped, axis=1, inplace=True)

# %%
# Target Encoding for CONTRACT_LINE_ITEMS
finalized_df_ohe_to_process["CONTRACT_LINE_ITEMS"].fillna("NA", inplace=True)
finalized_df_ohe_to_process["CONTRACT_LINE_ITEMS"] = finalized_df_ohe_to_process["CONTRACT_LINE_ITEMS"].str.replace(r"\\d+x ", "", regex=True)

for i, row in finalized_df_ohe_to_process.iterrows():
    t = row["CONTRACT_LINE_ITEMS"]
    arr = t.split("-")
    arr = [x.strip() for x in arr]
    arr_s = sorted(arr)
    key = "-".join([s for s in arr_s])
    finalized_df_ohe_to_process.loc[i, "CONTRACT_LINE_ITEMS"] = key

stage_path = "@PS_DOCUWARE_CHURN/ps_docuware_target_encoder_object"

X_train = finalized_df_ohe_to_process.copy()

import_dir = sys._xoptions.get("snowflake_import_directory")
enc_file = os.path.join('/tmp', 'ENC_CURRENT.joblib.gz')
session.file.get("@PS_DOCUWARE_CHURN/ps_docuware_target_encoder_object/ENC_CURRENT.joblib.gz", '/tmp')
enc = load(enc_file)

X_train = X_train.reindex(columns = enc.feature_names_in_, fill_value=0)
X_train_encoded = enc.transform(X_train)

# %%
# Imputation for Time Series columns
ts_columns = ["CUST_ACCOUNT_NUMBER", "YYYYWK", "DOCUMENTS_OPENED", "USED_STORAGE__MB", "INVOICE_REVLINE_TOTAL", "ORIGINAL_AMOUNT_DUE", "FUNCTIONAL_AMOUNT"]
raw_df_active["transformed_YYYYWK"] = raw_df_active["MONTH"].apply(convert_date_to_yyyywk)

#%%
# Impute missing YYYYWK with equivalent MONTH
raw_df_active["YYYYWK"].fillna(raw_df_active["transformed_YYYYWK"], inplace=True)
raw_df_active.drop("transformed_YYYYWK", axis=1, inplace=True)
ts_df = raw_df_active[ts_columns]
ts_df = ts_df[ts_df['YYYYWK'].notna()]
ts_df['YYYYWK'] = ts_df['YYYYWK'].astype(int)
ts_df['CUST_ACCOUNT_NUMBER'] = ts_df['CUST_ACCOUNT_NUMBER'].astype(int)
ts_df.rename(columns={'USED_STORAGE__MB':'USED_STORAGE_MB'},inplace=True)
ts_df_sorted = ts_df.sort_values(['CUST_ACCOUNT_NUMBER','YYYYWK']).drop_duplicates()
# %%
print(f"ts_df_sorted.shape: {ts_df_sorted.shape}")
print(ts_df_sorted.head().to_string())
# %%
only_features = ts_df_sorted.copy()
only_features = only_features.fillna(0)

#ts_comprehensive_df = session.sql("SELECT * FROM RAC_RAPID_DEV.RUS_AIML.PS_DOCUWARE_PREDICTION_TS_DF_ONLY").to_pandas()
#training_set_ts_df = session.sql("SELECT * FROM RAC_RAPID_DEV.RUS_AIML.PS_DOCUWARE_ONLY_TS_DF").to_pandas()
#ts_comprehensive_df_active = time_series_ts_fresh_features(only_features)
# %% [markdown]
# # Store Live Customers' TimeSeries data before TSFRESH into DB

# session.write_pandas(only_features, "PS_DOCUWARE_LIVE_RAW_DATA_BEFORE_TSFRESH", auto_create_table=True, overwrite = True)
# # %%
# /*WITH fi(CUST_ACCOUNT_NUMBER, YYYYWK) AS(
#     SELECT CUST_ACCOUNT_NUMBER, min(YYYYWK) AS YY FROM RUS_AIML.DOCUWARE_USAGE_JAPAN_V1_LATEST_V GROUP BY(CUST_ACCOUNT_NUMBER)
# )

# SELECT a.CUST_ACCOUNT_NUMBER, a.YYYYWK FROM fi a join RUS_AIML.PS_DOCUWARE_L1_CUST b on a.CUST_ACCOUNT_NUMBER = b.CUST_ACCOUNT_NUMBER where b.CHURNED_FLAG=False;*/
# %%
#all_usage = cell183.to_pandas()
# %%
#all_usage

# %%
#all_usage.loc[:'YYYYWK'] = 202336

# %%
#all_usage.head()

# %%
# """raw_df = session.sql("SELECT * FROM RAC_RAPID_DEV.RUS_AIML.PS_DOCUWARE_RAW_DATA_PREDICTION").to_pandas()
# # CK 5-21 not sure why extraction was used over prediction raw_df = session.sql("SELECT * FROM RAC_RAPID_DEV.RUS_AIML.PS_DOCUWARE_RAW_DATA_EXTRACTION").to_pandas()
# df_cust_earliest_date = raw_df[['CUST_ACCOUNT_NUMBER', 'FINAL_EARLIEST_DATE']]
# df_cust_earliest_date.drop_duplicates(inplace=True)
# df_cust_earliest_date['CUST_ACCOUNT_NUMBER'] = df_cust_earliest_date['CUST_ACCOUNT_NUMBER'].astype(int)
# all_usage['CUST_ACCOUNT_NUMBER'] = all_usage['CUST_ACCOUNT_NUMBER'].astype(int)
# df_merged = pd.merge(df_cust_earliest_date, all_usage , on='CUST_ACCOUNT_NUMBER', how='inner')"""
# %%
#raw_df.head()

# %%
#df_merged.head()

# %%
#final_features_for_FE

# %%
# """df_merged['CUST_ACCOUNT_NUMBER'] = df_merged['CUST_ACCOUNT_NUMBER'].astype(int)
# final_features_for_FE['CUST_ACCOUNT_NUMBER'] = final_features_for_FE['CUST_ACCOUNT_NUMBER'].astype(int)
# final_features_for_FE_trimmed = pd.DataFrame()

# for ind,row in df_merged.iterrows():
#     t = final_features_for_FE[ final_features_for_FE['CUST_ACCOUNT_NUMBER'] == row['CUST_ACCOUNT_NUMBER'] ]
#     t['YYYYWK'] = t['YYYYWK'].astype(int)
#     yyyywk_date = convert_yyyywk_to_actual_mid_date(row['YYYYWK'])

#     row['YYYYWK'] = int(row['YYYYWK'])
#     tt = t[ t['YYYYWK'] >= row['YYYYWK'] ]
#     tt['DAYS_TO_CHURN'] = tt['DAYS_TO_CHURN'] - (pd.to_datetime(yyyywk_date) - pd.to_datetime(row['FINAL_EARLIEST_DATE'])).days
#     final_features_for_FE_trimmed = pd.concat([tt, final_features_for_FE_trimmed], axis=0, ignore_index=True)"""

# %%
#final_features_for_FE_trimmed.head()

# %%
#session.write_pandas(final_features_for_FE_trimmed, "PS_DOCUWARE_LIVE_RAW_DATA_BEFORE_TSFRESH", auto_create_table=True, overwrite = True)
# %%
# Usage visualization
# Example 1: Linear Trend (Default)
# Create visualizer with linear trend
# Quick test to see if plots appear
# Create visualizer with moving average
viz_moving_avg = UsageVisualizer(
    figsize=(12, 8),
    start_date='2020-01-01',
    end_date='2025-08-08',
    trend_type='moving_avg',
    trend_window=8  # 8-week rolling average
)

# Run analysis
viz_moving_avg.plot_complete_analysis(
    usage_df=only_features,
    customer_df=l1_cust_df,
    save_dir=outputs / "plots" / 'usage_analysis_moving_avg'
)

# What you'll see:
# - Scatter plots with smooth dashed trend lines
# - Lines follow the data more closely than linear
# - Removes weekly spikes/dips to show underlying pattern
# %%
viz_moving_avg = UsageVisualizer(
    figsize=(12, 8),
    start_date='2020-01-01',
    end_date='2025-08-08',
    trend_type='linear',
    trend_window=8  # 8-week rolling average
)

# Run analysis
viz_moving_avg.plot_complete_analysis(
    usage_df=only_features,
    customer_df=l1_cust_df,
    save_dir=outputs / "plots" / 'usage_analysis_linear'
)
# %%
viz_moving_avg = UsageVisualizer(
    figsize=(12, 8),
    start_date='2020-01-01',
    end_date='2025-08-08',
    trend_type='lowess',
    trend_window=8  # 8-week rolling average
)

# Run analysis
viz_moving_avg.plot_complete_analysis(
    usage_df=only_features,
    customer_df=l1_cust_df,
    save_dir=outputs / "plots" / 'usage_analysis_lowes'
)
# %%
# Check for duplicate weeks in your aggregated data
weekly = only_features.groupby('YYYYWK').agg({
    'CUST_ACCOUNT_NUMBER': 'nunique'
}).reset_index()

# Check if there are duplicate YYYYWK values
duplicates = weekly[weekly.duplicated('YYYYWK', keep=False)]
print(f"Duplicate weeks found: {len(duplicates)}")
if len(duplicates) > 0:
    print(duplicates)

# Check the raw data for a specific week that shows this pattern
problem_week = 202420  # Pick a week where you see the double line
week_data = only_features[only_features['YYYYWK'] == problem_week]
print(f"\nData for week {problem_week}:")
print(f"Number of rows: {len(week_data)}")
print(f"Unique customers: {week_data['CUST_ACCOUNT_NUMBER'].nunique()}")

# Look at the actual values around the problematic area
weekly_sorted = weekly.sort_values('YYYYWK')
print("\nSample of weekly customer counts:")
print(weekly_sorted[['YYYYWK', 'CUST_ACCOUNT_NUMBER']].iloc[100:110])
# %%
# Zoom in on the y-axis to see the pattern better
viz = UsageVisualizer(
    figsize=(12, 8),
    start_date='2020-01-01',
    end_date='2025-08-08',
    trend_type='moving_avg'
)

# Modify the plotting method to set y-axis limits
# Or after plotting, zoom in:
fig, axes = plt.subplots(1, 1, figsize=(12, 4))
weekly = only_features.groupby('YYYYWK')['CUST_ACCOUNT_NUMBER'].nunique().reset_index()

axes.scatter(weekly['YYYYWK'], weekly['CUST_ACCOUNT_NUMBER'], alpha=0.6, s=30)
axes.set_ylim(250, 280)  # Zoom in to see the discrete values
axes.set_ylabel('Number of Active Customers')
axes.grid(True, alpha=0.3)
plt.show()
# %%
# Check the last few weeks of data
weekly = only_features.groupby('YYYYWK').agg({
    'DOCUMENTS_OPENED': 'sum',
    'USED_STORAGE_MB': 'sum',
    'CUST_ACCOUNT_NUMBER': 'nunique'
}).reset_index()

print("Last 5 weeks of data:")
print(weekly.tail())

# Check if last week has fewer days
last_week_data = only_features[only_features['YYYYWK'] == weekly['YYYYWK'].max()]
print(f"\nLast week ({weekly['YYYYWK'].max()}) has {len(last_week_data)} records")
print(f"Previous weeks average: {len(only_features) / weekly['YYYYWK'].nunique():.0f} records")
# %%
# Look for similar drops in previous years
weekly['year'] = weekly['YYYYWK'] // 100
weekly['week'] = weekly['YYYYWK'] % 100

# Check week 31-32 (early August) across years
august_weeks = weekly[weekly['week'].between(30, 35)]
print("Early August patterns across years:")
print(august_weeks[['YYYYWK', 'DOCUMENTS_OPENED', 'CUST_ACCOUNT_NUMBER']])
# %%
# Remove bad future data
only_features_clean = only_features[only_features['YYYYWK'] <= 202532]  # Up to week 32 of 2025

# Verify the cleaning
print(f"Original data: {len(only_features)} rows")
print(f"Clean data: {len(only_features_clean)} rows")
print(f"Removed: {len(only_features) - len(only_features_clean)} bad rows")

# Check the new last weeks
weekly_clean = only_features_clean.groupby('YYYYWK').agg({
    'DOCUMENTS_OPENED': 'sum',
    'USED_STORAGE_MB': 'sum',
    'CUST_ACCOUNT_NUMBER': 'nunique'
}).reset_index()

print("\nLast 5 weeks after cleaning:")
print(weekly_clean.tail())
# %% [markdown]
# # Feature Engineering using TSFRESH
# %%
# # CK 5-21 ts_comprehensive_df_active = engineer_timeseries_cols_using_tsfresh_for_live_customers(final_features_for_FE_trimmed)
# # CPU TIME: ~ 2 hrs
# # we may skip this step and instead read the saved featurres.
with timer():
    ts_comprehensive_df_active = engineer_timeseries_cols_using_tsfresh_for_live_customers(only_features)
# %%
ts_comprehensive_df_joblib_tsfresh_active = outputs / "serialized_features" / "tsfresh" / "ts_comprehensive_df_joblib_tsfresh_active.joblib.gz"
if write_to_outputs:
    JoblibSerializer().save_data(
        ts_comprehensive_df_active,
        ts_comprehensive_df_joblib_tsfresh_active
    )
# %%
# Justin review: Read ...
# We read it if we did not use tsfresh to save time
if read_from_outputs:
    ts_comprehensive_df_active_loaded = JoblibSerializer().load_data(ts_comprehensive_df_joblib_tsfresh_active)

ts_comprehensive_df_active.equals(ts_comprehensive_df_active_loaded)

# %%
ts_comprehensive_df_active = ts_comprehensive_df_active.replace([np.inf, -np.inf], 0)
ts_comprehensive_df_active = ts_comprehensive_df_active.fillna(0)
training_set_ts_df = ts_comprehensive_df_active.copy()
same_features = training_set_ts_df.columns
# %%
#same_features
#ts_comprehensive_df_active = fts_comprehensive_df_active
# %% [markdown]
# # Store LIVE Customers' TSFRESH time series generated features
# %%
ts_comprehensive_df_active.shape
# session.write_pandas(ts_comprehensive_df_active, "PS_DOCUWARE_LIVE_CUSTOMER_FEATURES", auto_create_table=True, overwrite = True)
# %%
# #comprehensive_imputed_df_active['FINAL_EARLIEST_DATE']
# temp = raw_df_active[['CUST_ACCOUNT_NUMBER', 'FINAL_EARLIEST_DATE']].drop_duplicates()
# temp.shape, ts_comprehensive_df_active.shape, X_train_encoded.shape
# %% [markdown]
# # Data preprocessing to prepare it for inferencing
# %%
ts_comprehensive_df_active = ts_comprehensive_df_active.reindex(columns=same_features, fill_value=0)
ts_comprehensive_df_active["CUST_ACCOUNT_NUMBER"] = ts_comprehensive_df_active["CUST_ACCOUNT_NUMBER"].astype(int)

X_train_encoded["CUST_ACCOUNT_NUMBER"] = X_train_encoded["CUST_ACCOUNT_NUMBER"].astype(int)
df_earliest_date = raw_df_active[['CUST_ACCOUNT_NUMBER', 'FINAL_EARLIEST_DATE']].drop_duplicates()
df_earliest_date["CUST_ACCOUNT_NUMBER"] = df_earliest_date["CUST_ACCOUNT_NUMBER"].astype(int)

comprehensive_imputed_df_active = pd.merge(ts_comprehensive_df_active, X_train_encoded, on="CUST_ACCOUNT_NUMBER", how="inner")
comprehensive_imputed_df_active = pd.merge(comprehensive_imputed_df_active, df_earliest_date, on="CUST_ACCOUNT_NUMBER", how="inner")
comprehensive_imputed_df_active['CREATION_DATE'] = pd.Timestamp.today().date()

#comprehensive_imputed_df_active['DAYS_ELAPSED_TILL_DATE'] = (pd.to_datetime(comprehensive_imputed_df_active['CREATION_DATE']) - pd.to_datetime(comprehensive_imputed_df_active['FINAL_EARLIEST_DATE'])).dt.days

#stratified_df_active = pd.DataFrame()

# Get CLUSTER Information in the dataframe
#label_count=1
# for i in range(365, 1461, 365):
#     XX = comprehensive_imputed_df_active[((i-365) < comprehensive_imputed_df_active["DAYS_ELAPSED_TILL_DATE"]) & (comprehensive_imputed_df_active["DAYS_ELAPSED_TILL_DATE"] <= i)]
#     #XX["SAMPLE"] = label_count
#     stratified_df_active = pd.concat([stratified_df_active, XX], axis=0, ignore_index=True)
#     #label_count+=1

#     if i==1460:
#         XX = comprehensive_imputed_df_active[1460 < comprehensive_imputed_df_active["DAYS_ELAPSED_TILL_DATE"]]
#         #XX["SAMPLE"] = label_count
#         stratified_df_active = pd.concat([stratified_df_active, XX], axis=0, ignore_index=True)
comprehensive_imputed_df_active_with_cluster = comprehensive_imputed_df_active.copy()
# comprehensive_imputed_df_active_with_cluster = stratified_df_active.copy()

for col in comprehensive_imputed_df_active_with_cluster.columns:
    if pd.api.types.is_sparse(comprehensive_imputed_df_active_with_cluster[col]):
        comprehensive_imputed_df_active_with_cluster[col] = comprehensive_imputed_df_active_with_cluster[col].sparse.to_dense()

#import_dir = sys._xoptions.get("snowflake_import_directory")
# model_file = os.path.join('/tmp', 'xgb.joblib.gz')
# session.file.get("@PS_DOCUWARE_CHURN/ps_docuware_churn_model/xgb.joblib.gz", "/tmp")
# best_model = load(model_file)

#input_features = comprehensive_imputed_df_active_with_cluster.drop(["CUST_ACCOUNT_NUMBER", "CREATION_DATE", "DAYS_ELAPSED_TILL_DATE"], axis=1)
input_features = comprehensive_imputed_df_active_with_cluster.drop(["CUST_ACCOUNT_NUMBER", "CREATION_DATE"], axis=1)

input_features = convert_to_float(input_features)
# input_features = input_features.reindex(columns=best_model.feature_names_in_, fill_value=0)
input_features = input_features.reindex(columns=best_model.feature_names_, fill_value=0)
# %%
input_features
# %% [markdown]
# # Store features of Live Customers into Staging
# %%
df_live_customers = session.create_dataframe(input_features)
#df_live_customers.write.copy_into_location("@PS_DOCUWARE_CHURN/df_live_customers_21Apr2025.csv", file_format_type="csv", format_type_options={"COMPRESSION": "NONE"}, header=True, overwrite=True )
# %% [markdown]
# # Store features of Live Customers into DB
# %%
df_live_customers
# %%
df_live_customers = df_live_customers.to_pandas()
# %%
type(df_live_customers)
# %%
# session.write_pandas(df_live_customers, "PS_DOCUWARE_LIVE_CUSTOMER_FEATURES", auto_create_table=True, overwrite = True)
PS_DOCUWARE_LIVE_CUSTOMER_FEATURES = outputs / "serialized_features" / "tsfresh" / "PS_DOCUWARE_LIVE_CUSTOMER_FEATURES.joblib.gz"
if write_to_outputs:
    saved_path = JoblibSerializer().save_data(
        df_live_customers,
        PS_DOCUWARE_LIVE_CUSTOMER_FEATURES
    )
# %%
#session.write_pandas(df_live_customers, "PS_DOCUWARE_LIVE_CUSTOMER_FEATURES", auto_create_table=True, overwrite = True)
# %%
import shap
# %% [markdown]
# # Inferencing for Live customers to predict the churn duration
# %%
y_pred_transformed = best_model.predict(input_features)
remaining_months = y_pred_transformed/30
#lifespan_months = np.exp(y_pred_transformed)

final_predicted_df = pd.DataFrame({
    "CUST_ACCOUNT_NUMBER": comprehensive_imputed_df_active["CUST_ACCOUNT_NUMBER"].astype(str),
    "MONTHS_REMAINING": remaining_months
})
# %%
final_predicted_df.head(50)
# %%
raw_df_active_1 = raw_df_active[['CUST_ACCOUNT_NUMBER', 'FINAL_EARLIEST_DATE']].drop_duplicates()

final_predicted_df = final_predicted_df.merge(raw_df_active_1, on='CUST_ACCOUNT_NUMBER', how='inner')

final_predicted_df['CREATION_DATE'] = pd.Timestamp.today().date()
final_predicted_df = final_predicted_df.rename({'FINAL_EARLIEST_DATE':'EARLIEST_START_DATE'})


final_predicted_df['DAYS_ELAPSED'] = (pd.to_datetime(final_predicted_df['CREATION_DATE']) - pd.to_datetime(final_predicted_df['FINAL_EARLIEST_DATE'])).dt.days
final_predicted_df['MONTHS_ELAPSED'] = (pd.to_datetime(final_predicted_df['CREATION_DATE']) - pd.to_datetime(final_predicted_df['FINAL_EARLIEST_DATE'])).dt.days/30
#final_predicted_df['MONTHS_REMAINING'] = final_predicted_df['LIFESPAN_MONTHS'] #- final_predicted_df['MONTHS_ELAPSED']
final_predicted_df['LIFESPAN_MONTHS'] = final_predicted_df['MONTHS_REMAINING'] + final_predicted_df['MONTHS_ELAPSED']
final_predicted_df.head(100)

# %%
print(final_predicted_df.head().to_string())
# %%
"""
Standalone prediction analysis - no external dependencies
Works directly with your final_predicted_df
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'seaborn-darkgrid')

def analyze_churn_predictions(final_predicted_df, save_plots=True, output_dir='outputs/predictions_analysis'):
    """
    Complete analysis of your churn predictions with visualizations
    """

    # Create output directory if saving
    if save_plots:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CHURN PREDICTION ANALYSIS")
    print("="*70)

    # 1. Basic Statistics
    print("\n📊 BASIC STATISTICS:")
    print("-"*50)

    total = len(final_predicted_df)
    print(f"Total customers: {total:,}")
    print(f"Analysis date: {final_predicted_df['CREATION_DATE'].iloc[0] if 'CREATION_DATE' in final_predicted_df.columns else 'N/A'}")

    print(f"\nMONTHS_REMAINING Statistics:")
    print(f"  Mean: {final_predicted_df['MONTHS_REMAINING'].mean():.2f} months")
    print(f"  Median: {final_predicted_df['MONTHS_REMAINING'].median():.2f} months")
    print(f"  Std Dev: {final_predicted_df['MONTHS_REMAINING'].std():.2f} months")
    print(f"  Min: {final_predicted_df['MONTHS_REMAINING'].min():.2f} months")
    print(f"  Max: {final_predicted_df['MONTHS_REMAINING'].max():.2f} months")

    # 2. Risk Assessment
    print("\n⚠️ RISK ASSESSMENT:")
    print("-"*50)

    # Calculate risk buckets
    immediate_risk = (final_predicted_df['MONTHS_REMAINING'] < 1).sum()
    high_risk = ((final_predicted_df['MONTHS_REMAINING'] >= 1) &
                 (final_predicted_df['MONTHS_REMAINING'] < 3)).sum()
    medium_risk = ((final_predicted_df['MONTHS_REMAINING'] >= 3) &
                   (final_predicted_df['MONTHS_REMAINING'] < 6)).sum()
    low_risk = ((final_predicted_df['MONTHS_REMAINING'] >= 6) &
                (final_predicted_df['MONTHS_REMAINING'] < 12)).sum()
    very_low_risk = (final_predicted_df['MONTHS_REMAINING'] >= 12).sum()

    print(f"Immediate (<1 month):  {immediate_risk:,} ({immediate_risk/total*100:.1f}%)")
    print(f"High Risk (1-3 months): {high_risk:,} ({high_risk/total*100:.1f}%)")
    print(f"Medium Risk (3-6 months): {medium_risk:,} ({medium_risk/total*100:.1f}%)")
    print(f"Low Risk (6-12 months): {low_risk:,} ({low_risk/total*100:.1f}%)")
    print(f"Very Low Risk (>12 months): {very_low_risk:,} ({very_low_risk/total*100:.1f}%)")

    # 3. Diagnosis
    print("\n🔍 DIAGNOSIS:")
    print("-"*50)

    risk_3m = (immediate_risk + high_risk) / total * 100
    risk_6m = (immediate_risk + high_risk + medium_risk) / total * 100

    if risk_3m > 30:
        print(f"🔴 CRITICAL: {risk_3m:.1f}% predicted to churn within 3 months")
        print("   This is unusually high and suggests model calibration issues")
    elif risk_3m > 20:
        print(f"🟠 WARNING: {risk_3m:.1f}% predicted to churn within 3 months")
        print("   This is higher than typical (~10-15%)")
    elif risk_3m > 10:
        print(f"🟡 MODERATE: {risk_3m:.1f}% predicted to churn within 3 months")
        print("   This is within reasonable range but monitor closely")
    else:
        print(f"🟢 HEALTHY: {risk_3m:.1f}% predicted to churn within 3 months")
        print("   Predictions appear reasonable")

    # Annual churn projection
    annual_churn = (final_predicted_df['MONTHS_REMAINING'] <= 12).sum() / total * 100
    print(f"\n📅 Projected annual churn rate: {annual_churn:.1f}%")

    # 4. Visualizations
    print("\n📈 GENERATING VISUALIZATIONS...")
    print("-"*50)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))

    # 1. Distribution Histogram
    ax1 = plt.subplot(3, 3, 1)
    n, bins, patches = ax1.hist(final_predicted_df['MONTHS_REMAINING'], bins=30,
                                edgecolor='black', alpha=0.7, color='skyblue')

    # Color code by risk
    for i, patch in enumerate(patches):
        if bins[i] < 3:
            patch.set_facecolor('#e74c3c')  # Red
        elif bins[i] < 6:
            patch.set_facecolor('#f39c12')  # Orange
        elif bins[i] < 12:
            patch.set_facecolor('#f1c40f')  # Yellow
        else:
            patch.set_facecolor('#2ecc71')  # Green

    ax1.axvline(final_predicted_df['MONTHS_REMAINING'].mean(), color='red',
               linestyle='--', linewidth=2, label=f'Mean: {final_predicted_df["MONTHS_REMAINING"].mean():.1f}')
    ax1.axvline(final_predicted_df['MONTHS_REMAINING'].median(), color='green',
               linestyle='--', linewidth=2, label=f'Median: {final_predicted_df["MONTHS_REMAINING"].median():.1f}')
    ax1.set_xlabel('Months Remaining')
    ax1.set_ylabel('Number of Customers')
    ax1.set_title('Distribution of Predicted Remaining Lifetime')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative Distribution
    ax2 = plt.subplot(3, 3, 2)
    sorted_months = np.sort(final_predicted_df['MONTHS_REMAINING'])
    cumulative = np.arange(1, len(sorted_months) + 1) / len(sorted_months) * 100
    ax2.plot(sorted_months, cumulative, linewidth=2, color='#3498db')
    ax2.fill_between(sorted_months, cumulative, alpha=0.3, color='#3498db')
    ax2.axvline(3, color='red', linestyle=':', alpha=0.5, label='3 months')
    ax2.axvline(6, color='orange', linestyle=':', alpha=0.5, label='6 months')
    ax2.axvline(12, color='yellow', linestyle=':', alpha=0.5, label='12 months')
    ax2.axhline(50, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Months Remaining')
    ax2.set_ylabel('Cumulative % of Customers')
    ax2.set_title('Cumulative Churn Timeline')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Risk Distribution Pie Chart
    ax3 = plt.subplot(3, 3, 3)
    risk_labels = ['Immediate\n<1mo', 'High\n1-3mo', 'Medium\n3-6mo', 'Low\n6-12mo', 'Very Low\n>12mo']
    risk_sizes = [immediate_risk, high_risk, medium_risk, low_risk, very_low_risk]
    risk_colors = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#2ecc71']

    # Only plot non-zero categories
    non_zero_labels = [label for label, size in zip(risk_labels, risk_sizes) if size > 0]
    non_zero_sizes = [size for size in risk_sizes if size > 0]
    non_zero_colors = [color for color, size in zip(risk_colors, risk_sizes) if size > 0]

    wedges, texts, autotexts = ax3.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors,
                                        autopct='%1.1f%%', startangle=90)
    ax3.set_title('Customer Risk Distribution')

    # 4. Box Plot
    ax4 = plt.subplot(3, 3, 4)
    bp = ax4.boxplot([final_predicted_df['MONTHS_REMAINING']], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][0].set_alpha(0.7)
    ax4.set_ylabel('Months Remaining')
    ax4.set_title('Lifecycle Distribution - Box Plot')
    ax4.grid(True, alpha=0.3)

    # Add quartile annotations
    q1 = final_predicted_df['MONTHS_REMAINING'].quantile(0.25)
    q2 = final_predicted_df['MONTHS_REMAINING'].quantile(0.50)
    q3 = final_predicted_df['MONTHS_REMAINING'].quantile(0.75)
    ax4.text(1.1, q1, f'Q1: {q1:.1f}', va='center')
    ax4.text(1.1, q2, f'Median: {q2:.1f}', va='center')
    ax4.text(1.1, q3, f'Q3: {q3:.1f}', va='center')

    # 5. Tenure vs Remaining Lifetime (if tenure data exists)
    ax5 = plt.subplot(3, 3, 5)
    if 'MONTHS_ELAPSED' in final_predicted_df.columns:
        ax5.scatter(final_predicted_df['MONTHS_ELAPSED'],
                   final_predicted_df['MONTHS_REMAINING'],
                   alpha=0.5, s=20, c=final_predicted_df['MONTHS_REMAINING'],
                   cmap='RdYlGn', vmin=0, vmax=24)
        ax5.set_xlabel('Customer Tenure (Months Elapsed)')
        ax5.set_ylabel('Predicted Months Remaining')
        ax5.set_title('Tenure vs Predicted Remaining Lifetime')

        # Add trend line
        z = np.polyfit(final_predicted_df['MONTHS_ELAPSED'],
                      final_predicted_df['MONTHS_REMAINING'], 1)
        p = np.poly1d(z)
        ax5.plot(final_predicted_df['MONTHS_ELAPSED'],
                p(final_predicted_df['MONTHS_ELAPSED']),
                "r--", alpha=0.8, linewidth=2)

        correlation = final_predicted_df['MONTHS_ELAPSED'].corr(final_predicted_df['MONTHS_REMAINING'])
        ax5.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=ax5.transAxes, va='top')
        ax5.grid(True, alpha=0.3)
        plt.colorbar(ax5.collections[0], ax=ax5, label='Months Remaining')
    else:
        ax5.text(0.5, 0.5, 'No tenure data available',
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Tenure Analysis Not Available')

    # 6. Monthly Churn Projection
    ax6 = plt.subplot(3, 3, 6)

    # Create monthly buckets for next 12 months
    monthly_churn = []
    for month in range(1, 13):
        month_start = month - 1
        month_end = month
        churning = ((final_predicted_df['MONTHS_REMAINING'] > month_start) &
                   (final_predicted_df['MONTHS_REMAINING'] <= month_end)).sum()
        monthly_churn.append(churning)

    months = list(range(1, 13))
    bars = ax6.bar(months, monthly_churn, color='#3498db', edgecolor='black', alpha=0.7)

    # Color code bars
    for i, bar in enumerate(bars):
        if i < 3:
            bar.set_facecolor('#e74c3c')
        elif i < 6:
            bar.set_facecolor('#f39c12')
        else:
            bar.set_facecolor('#2ecc71')

    ax6.set_xlabel('Month')
    ax6.set_ylabel('Number of Customers Churning')
    ax6.set_title('Monthly Churn Projection (Next 12 Months)')
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Density Plot
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(final_predicted_df['MONTHS_REMAINING'], bins=50, density=True,
             alpha=0.7, color='skyblue', edgecolor='black')

    # Fit and plot normal distribution
    mu, std = stats.norm.fit(final_predicted_df['MONTHS_REMAINING'])
    xmin, xmax = ax7.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax7.plot(x, p, 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.1f}, σ={std:.1f}')

    ax7.set_xlabel('Months Remaining')
    ax7.set_ylabel('Density')
    ax7.set_title('Distribution with Normal Fit')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Q-Q Plot for normality check
    ax8 = plt.subplot(3, 3, 8)
    stats.probplot(final_predicted_df['MONTHS_REMAINING'], dist="norm", plot=ax8)
    ax8.set_title('Q-Q Plot (Normality Check)')
    ax8.grid(True, alpha=0.3)

    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')

    # Create summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Total Customers', f'{total:,}'],
        ['Mean Remaining', f'{final_predicted_df["MONTHS_REMAINING"].mean():.2f} months'],
        ['Median Remaining', f'{final_predicted_df["MONTHS_REMAINING"].median():.2f} months'],
        ['3-Month Risk', f'{risk_3m:.1f}%'],
        ['6-Month Risk', f'{risk_6m:.1f}%'],
        ['Annual Churn', f'{annual_churn:.1f}%'],
        ['Std Deviation', f'{final_predicted_df["MONTHS_REMAINING"].std():.2f}'],
        ['CV', f'{final_predicted_df["MONTHS_REMAINING"].std()/final_predicted_df["MONTHS_REMAINING"].mean():.2f}']
    ]

    table = ax9.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style the header row
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code risk rows
    table[(4, 1)].set_facecolor('#ffcccc' if risk_3m > 30 else '#ccffcc')
    table[(5, 1)].set_facecolor('#ffcccc' if risk_6m > 50 else '#ccffcc')

    ax9.set_title('Summary Statistics')

    plt.suptitle('Churn Prediction Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_plots:
        plt.savefig(output_path / 'churn_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved to: {output_path / 'churn_analysis_dashboard.png'}")

    plt.show()

    # Return statistics
    return {
        'total_customers': total,
        'risk_3m_pct': risk_3m,
        'risk_6m_pct': risk_6m,
        'annual_churn_pct': annual_churn,
        'mean_months': final_predicted_df['MONTHS_REMAINING'].mean(),
        'median_months': final_predicted_df['MONTHS_REMAINING'].median(),
        'std_months': final_predicted_df['MONTHS_REMAINING'].std()
    }


def create_monthly_projection(final_predicted_df, end_date='2025-12-31', save_path=None):
    """
    Create monthly churn projection through end_date
    """

    # Convert to datetime
    if 'CREATION_DATE' in final_predicted_df.columns:
        start_date = pd.to_datetime(final_predicted_df['CREATION_DATE'].iloc[0])
    else:
        start_date = pd.Timestamp.today()

    end_date = pd.to_datetime(end_date)

    # Create month range
    month_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Calculate predicted churn date for each customer
    final_predicted_df['PREDICTED_CHURN_DATE'] = start_date + pd.to_timedelta(
        final_predicted_df['MONTHS_REMAINING'] * 30, unit='D'
    )

    # Count by month
    monthly_data = []
    total_customers = len(final_predicted_df)

    for month_start in month_range:
        month_end = month_start + pd.offsets.MonthEnd(0)

        # Customers churning this month
        churning = ((final_predicted_df['PREDICTED_CHURN_DATE'] >= month_start) &
                   (final_predicted_df['PREDICTED_CHURN_DATE'] <= month_end)).sum()

        # Still active at month end
        active = (final_predicted_df['PREDICTED_CHURN_DATE'] > month_end).sum()

        # Already churned
        already_churned = (final_predicted_df['PREDICTED_CHURN_DATE'] < month_start).sum()

        monthly_data.append({
            'Month': month_start.strftime('%Y-%m'),
            'Churning': churning,
            'Active': active,
            'Already_Churned': already_churned,
            'Total': total_customers
        })

    monthly_df = pd.DataFrame(monthly_data)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Stacked bar chart
    x = range(len(monthly_df))
    width = 0.8

    p1 = ax1.bar(x, monthly_df['Active'], width, label='Active', color='#2ecc71', alpha=0.8)
    p2 = ax1.bar(x, monthly_df['Churning'], width, bottom=monthly_df['Active'],
                label='Churning This Month', color='#e74c3c', alpha=0.8)
    p3 = ax1.bar(x, monthly_df['Already_Churned'], width,
                bottom=monthly_df['Active'] + monthly_df['Churning'],
                label='Already Churned', color='#95a5a6', alpha=0.6)

    ax1.set_ylabel('Number of Customers')
    ax1.set_title('Monthly Customer Lifecycle Projection')
    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly_df['Month'], rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (churning, active) in enumerate(zip(monthly_df['Churning'], monthly_df['Active'])):
        if churning > 0:
            ax1.text(i, active + churning/2, str(int(churning)),
                    ha='center', va='center', fontweight='bold', color='white')

    # Churn rate line
    monthly_df['Churn_Rate'] = monthly_df['Churning'] / monthly_df['Total'] * 100
    ax2.plot(x, monthly_df['Churn_Rate'], 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax2.fill_between(x, monthly_df['Churn_Rate'], alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Monthly Churn Rate (%)')
    ax2.set_title('Monthly Churn Rate Trend')
    ax2.set_xticks(x)
    ax2.set_xticklabels(monthly_df['Month'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # Add average line
    avg_rate = monthly_df['Churn_Rate'].mean()
    ax2.axhline(y=avg_rate, color='red', linestyle='--', alpha=0.5,
               label=f'Average: {avg_rate:.1f}%')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Monthly projection saved to: {save_path}")

    plt.show()

    return monthly_df
# %% [markdown]
stats = analyze_churn_predictions(final_predicted_df, save_plots=True)

monthly_df = create_monthly_projection(final_predicted_df, end_date='2025-12-31')
# %%
total_in_usage = len(usageFIX['CUST_ACCOUNT_NUMBER'].unique())
active_in_usage = len(usageFIX[usageFIX['CHURNED_FLAG'] == 'N']['CUST_ACCOUNT_NUMBER'].unique())
churned_in_usage = len(usageFIX[usageFIX['CHURNED_FLAG'] =='Y']['CUST_ACCOUNT_NUMBER'].unique())

# %%
# # Store Prediction for Live customers into DB
# %%
PS_DOCUWARE_PREDICTION_DATA_NEW = outputs / "serialized_data" / "predictions" / "PS_DOCUWARE_PREDICTION_DATA_NEW.joblib.gz"
# session.write_pandas(final_predicted_df, "PS_DOCUWARE_PREDICTION_DATA_NEW", auto_create_table=True, overwrite = True)
if write_to_outputs:
    saved_path = JoblibSerializer().save_data(
        final_predicted_df,
        PS_DOCUWARE_PREDICTION_DATA_NEW
    )

print(f"Total: {total_in_usage:,}")
print(f"Active: {active_in_usage:,} ({active_in_usage/total_in_usage*100:.1f}%)")
print(f"Churned: {churned_in_usage:,} ({churned_in_usage/total_in_usage*100:.1f}%)")

# %%
 # Calculate ACTUAL historical churn rate using l1_cust_df
  print("\n" + "="*60)
  print("ACTUAL HISTORICAL CHURN RATE FROM L1_CUST")
  print("="*60)

  # Overall churn rate from customer master table
  total_customers = len(l1_cust_df)
  churned_customers = len(l1_cust_df[l1_cust_df['CHURNED_FLAG'] == 'Y'])
  active_customers = len(l1_cust_df[l1_cust_df['CHURNED_FLAG'] == 'N'])

  overall_churn_rate = (churned_customers / total_customers) * 100

  print(f"Total customers: {total_customers:,}")
  print(f"Active customers: {active_customers:,} ({active_customers/total_customers*100:.1f}%)")
  print(f"Churned customers: {churned_customers:,}({churned_customers/total_customers*100:.1f}%)")

  # Analyze churn dates if available
  if 'CHURN_DATE' in l1_cust_df.columns:
      churned_with_dates = l1_cust_df[l1_cust_df['CHURNED_FLAG'] == 'Y'].copy()
      churned_with_dates['CHURN_DATE'] = pd.to_datetime(churned_with_dates['CHURN_DATE'])

      # Remove any null churn dates
      churned_with_dates = churned_with_dates[churned_with_dates['CHURN_DATE'].notna()]

      if len(churned_with_dates) > 0:
          # Calculate time span
          earliest_churn = churned_with_dates['CHURN_DATE'].min()
          latest_churn = churned_with_dates['CHURN_DATE'].max()
          months_span = (latest_churn - earliest_churn).days / 30.44

          print(f"\nChurn date analysis:")
          print(f"  Earliest churn: {earliest_churn.date()}")
          print(f"  Latest churn: {latest_churn.date()}")
          print(f"  Time span: {months_span:.1f} months")

          # Monthly churn rate
          if months_span > 0:
              avg_monthly_churn = churned_customers / months_span
              avg_monthly_churn_rate = (avg_monthly_churn / total_customers) * 100
              print(f"  Average monthly churn: {avg_monthly_churn:.0f} customers")
              print(f"  Average monthly churn rate: {avg_monthly_churn_rate:.2f}%")

              # Annualized
              annual_churn_rate = avg_monthly_churn_rate * 12
              print(f"  Annualized churn rate: {annual_churn_rate:.1f}%")

  print("\n" + "="*60)
  print("COMPARISON WITH MODEL PREDICTIONS")
  print("="*60)
  print(f"Your model predicts: 6.5% monthly churn (~55% annual)")
  if 'avg_monthly_churn_rate' in locals():
      print(f"Actual historical: {avg_monthly_churn_rate:.2f}% monthly 
  ({avg_monthly_churn_rate*12:.1f}% annual)")
      if avg_monthly_churn_rate > 0:
          overestimation = 6.5 / avg_monthly_churn_rate
          print(f"Model overestimation factor: {overestimation:.1f}x")
          if overestimation > 3:
              print("🔴 CRITICAL: Model is severely overestimating churn!")
          elif overestimation > 2:
              print("🟠 WARNING: Model is significantly overestimating churn")
  else:
      print("Could not calculate monthly rate - check CHURN_DATE column")

# changed != to =
# session.sql("INSERT INTO rus_aiml.PS_DOCUWARE_PREDICTION_DATA_ARCHIVE SELECT * FROM rus_aiml.PS_DOCUWARE_PREDICTION_DATA_NEW WHERE CREATION_DATE not in (select creation_date from rus_aiml.PS_DOCUWARE_PREDICTION_DATA_ARCHIVE)")
# old sql INSERT INTO PS_DOCUWARE_PREDICTION_DATA_ARCHIVE SELECT * FROM PS_DOCUWARE_PREDICTION_DATA_NEW WHERE CREATION_DATE = CURRENT_DATE()
# %% [markdown]
# # Shap analysis
# %%
# !pip install shap
# %%
# # Inferencing
# y_pred = best_models.predict(test_25_X)
# remaining_months = y_pred/30
# print(remaining_months.shape)

# actual_remaining = test_25_y/30
# final_test_predicted_df = pd.DataFrame({
#     "CUST_ACCOUNT_NUMBER": last_rows_X_test.loc[test_25_X.index,"CUST_ACCOUNT_NUMBER"].astype(str),
#     "PREDICTED_MONTHS_REMAINING": remaining_months,
#     "ACTUAL_MONTH_REMAINING": actual_remaining,
#     "RESIDUAL": abs(remaining_months-actual_remaining)
# })

# final_test_predicted_df.head(50)

# %%
model_file = os.path.join('/tmp', 'xgb.joblib.gz')
session.file.get("@PS_DOCUWARE_CHURN/ps_docuware_churn_model/xgb.joblib.gz", "/tmp")
best_model = load(model_file)
#live_features = session.sql("SELECT * FROM RUS_AIML.PS_DOCUWARE_LIVE_CUSTOMER_FEATURES").to_pandas()
explainer = shap.Explainer(best_model.predict, test_25_X)
shap_values = explainer.shap_values(test_25_X)
# %%
shap_values.shape
# %%
shap.summary_plot(shap_values)
# %%
shap.summary_plot(shap_values, plot_type='violin')
# %%
feature_importance = global_shap_importance(best_models, test_25_X)
# %%
feature_importance_top40 = feature_importance.head(40)
# %% [markdown]
# # Plot of Shap to get top 40 important features for Live customer's churn prediction
# %%
# Sample data
y = feature_importance_top40['importance']
x = feature_importance_top40['features']

# Create a figure and a set of subplots with a specified size
fig, ax = plt.subplots(figsize=(40, 30))  # width=12 inches, height=8 inches
# Plot the data on the axes
ax.plot(x, y)
# Add labels and title
ax.set_xlabel('importance score')
ax.set_ylabel('Features')
ax.set_title('Graph of Top 40 Feature vs Importance While Predicting Input of 405 customers')
plt.xticks(x, x, rotation=45, ha='right')
#plt.savefig('C:/My Documents/work/Projects/Short Term Goal 2 - Customer Churn Prediction/ML Modeling Phase/Shap_Top_30_Importance_Features.png')
# Show the plot
plt.show()
# %%
test_25_index = test_25_X.index
# %%
test_25_X.shape
# %%
test_25_index
# %%
for ind,row in test_25_X.iterrows():
    print(ind)
    temp = pd.DataFrame([row])
    print(temp.shape)
    shap_values = explainer.shap_values(temp)
    shap.summary_plot(shap_values, plot_type='violin')

# %%
for ind,row in test_25_X.iterrows():
    print(ind)
    temp = pd.DataFrame([row])
    print(temp.shape)
    feature_importance = global_shap_importance(best_models, temp)

    feature_importance_top40 = feature_importance.head(40)

    # Sample data
    y = feature_importance_top40['importance']
    x = feature_importance_top40['features']

    # Create a figure and a set of subplots with a specified size
    fig, ax = plt.subplots(figsize=(40, 30))  # width=12 inches, height=8 inches

    # Plot the data on the axes
    ax.plot(x, y)

    # Add labels and title
    ax.set_xlabel('importance score')
    ax.set_ylabel('Features')
    ax.set_title('Graph of Top 40 Feature vs Importance While Predicting Input of 405 customers')
    plt.xticks(x, x, rotation=45, ha='right')
    #plt.savefig('C:/My Documents/work/Projects/Short Term Goal 2 - Customer Churn Prediction/ML Modeling Phase/Shap_Top_30_Importance_Features.png')
    # Show the plot
    plt.show()


# %%
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 3, 5]

plt.plot(x, y, marker='o', linestyle='-', color='b', label='Data Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('My Graph')
plt.legend()

graph_file = os.path.join("/tmp", "graph_new.jpg")
plt.savefig(graph_file)
session.file.put(graph_file, "@PS_DOCUWARE_CHURN/graph_new.jpg",overwrite=True)

plt.show()
# %%
 PUT /tmp/graph_new.jpg @PS_DOCUWARE_CHURN/ OVERWRITE=TRUE;
1+1


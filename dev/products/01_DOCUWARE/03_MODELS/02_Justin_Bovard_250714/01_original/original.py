# Advisor: Justin Bovard
# email: Justin.Bovard@ricoh-usa.com
# -----------------------------------------------------------------------------
# UPDATED ON: 2025-08-07
# CREATED ON: 2025-06-27
# -----------------------------------------------------------------------------
# COPYRIGHT@2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh
# -----------------------------------------------------------------------------
%load_ext autoreload
%autoreload 2
%load_ext watermark
%watermark
%matplotlib inline
# %%
# from snowflake.snowpark.context import get_active_session
# session = get_active_session()
# %%
# !pip install numpy==1.23.5 xgboost pandas==1.5.3 joblib==1.3.2 scikit-learn==1.3.2 tsfresh==0.20.1 category_encoders==2.6.2 scikit-optimize==0.9.0 statsmodels==0.13.5 scipy==1.10.1 tqdm==4.66.1 distributed==2023.3.1 dask==2023.3.1 shap
# %% [markdown]
# # Import all required libraries
# %%
import time
import math
import pandas as pd
import numpy as np
import re
import os
import joblib
import sys
import sklearn
import skopt
import shap
import category_encoders
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from datetime import datetime, timedelta
import snowflake.snowpark as snowpark
from snowflake.snowpark import Session
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import catboost as cb
import xgboost as xgb

import tsfresh
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit

from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series

from churn_aiml.utils.profiling import timer

from churn_aiml.utils.profiling import timer

import warnings
warnings.filterwarnings("ignore")
# %%
# print("Python version",sys.version)
# print("Pandas version", pd.__version__)
# print("Numpy version", np.__version__)
# print("Scikit-Learn version", sklearn.__version__)
# print("Scikit-Optimize version", skopt.__version__)
# print("XGBoost version", xgb.__version__)
# print("Category Encoders version", category_encoders.__version__)
# print("Joblib version", joblib.__version__)
# print("Ts fresh version", tsfresh.__version__)
# Print platform parameters
print('Versions control:')
%watermark --iversions
# %%
# connection_parameters = {
#             'user':'INTAIMLGEN',
#             'password':'Qpalzmbchdeula7#',
#             'account':'ricohusa.us-east-1.privatelink',
#             'warehouse':'RAC_AIML_NONPROD',
#             'database' : 'RAC_RAPID_UAT',
# }
# session = Session.builder.configs(connection_parameters).create()
# session.use_schema("RUS_AIML")
# session.use_role("RAC_UAT_AIML_ROLE")
# %%
    # account: "ricohusa.us-east-1.privatelink"
    # user: "INTAIMLGEN"
    # password: "QPalzmbchdeul77##"
    # warehouse: "RAC_AIML_NONPROD"
    # database: "RAC_RAPID_DEV"
    # schema: "RUS_AIML"
    # role: "RAC_DEV_AIML_ROLE"

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
# %%
# from snowflake.snowpark import Session
# from snowflake.snowpark.functions import col, regexp_replace, trim, to_char, rank, when_matched
# from snowflake.snowpark.window import Window
# from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType
# import snowflake.snowpark.functions as F
# %% [markdown]
# # UTILITY Function 1
# %%
def convert_to_float(df):
    return df.apply(pd.to_numeric, errors='coerce').astype(np.float32)

# %% [markdown]
# # UTILITY Function 2
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
def get_days_diff(yyyywk, last_yyyywk):
    curr = pd.to_datetime(convert_yyyywk_to_actual_mid_date(yyyywk))
    last = pd.to_datetime(convert_yyyywk_to_actual_mid_date(last_yyyywk))
    #print(curr, last)
    days_diff = (last- curr) / np.timedelta64(1, 'D')
    return days_diff
# %%
# print(yyyywk, last_yyyywk, days_diff)
get_days_diff(202318, 202401)
# %% [markdown]
# # UTILITY Function 3
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

# %%
#ts_comprehensive_df, ts_age_df = engineer_timeseries_cols_using_tsfresh(final_features_for_FE)
#ts_comprehensive_df_sf = session.create_dataframe(ts_comprehensive_df)
#ts_comprehensive_df_sf.write.copy_into_location("@PS_DOCUWARE_CHURN/TSFRESH/all_feature_TSFRESH/ts_comprehensive_df.csv", file_format_type="csv", format_type_options={"COMPRESSION": "NONE"}, header=True, overwrite=True )

#session.write_pandas(ts_comprehensive_df, "PS_DOCUWARE_TSFRESH_FOR_TRAINING", auto_create_table=True, overwrite = True)
#session.write_pandas(ts_age_df, "PS_DOCUWARE_TSFRESH_AGE", auto_create_table=True, overwrite = True)

# %%
#ts_comprehensive_df.shape, ts_age_df.shape

# %%
#ts_age_df.head()

# %% [markdown]
# # UTILITY Function 6
# %%
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
# #### Raw Data Extraction
# CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_PAYMENTS_V AS SELECT CUSTOMER_NO, RECEIPT_DATE, FUNCTIONAL_AMOUNT FROM RUS_AIML.PS_DOCUWARE_PAYMENTS;

# CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_REVENUE_V AS SELECT CUST_ACCOUNT_NUMBER, DATE_INVOICE_GL_DATE, INVOICE_REVLINE_TOTAL FROM RUS_AIML.PS_DOCUWARE_REVENUE;

# CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_TRX_V AS SELECT ACCOUNT_NUMBER, TRX_DATE, ORIGINAL_AMOUNT_DUE FROM RUS_AIML.PS_DOCUWARE_TRX;

# CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_CONTRACTS_SUBLINE_V AS SELECT CUST_ACCOUNT_NUMBER, SLINE_START_DATE, SLINE_END_DATE, SLINE_STATUS FROM RUS_AIML.PS_DOCUWARE_CONTRACT_SUBLINE;

# CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_RENEWALS_V AS SELECT TO_CHAR(BILLTOCUSTOMERNUMBER) AS BILLTOCUSTOMERNUMBER, TO_CHAR(SHIPTOCUSTNUM) AS SHIPTOCUSTNUM, STARTDATECOVERAGE, CONTRACT_END_DATE FROM RUS_AIML.PS_DOCUWARE_SSCD_RENEWALS;

# CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_L1_CUST_V AS SELECT CUST_ACCOUNT_NUMBER, CUST_PARTY_NAME, L3_RISE_CONSOLIDATED_NUMBER, L3_RISE_CONSOLIDATED_NAME, L2_RISE_CONSOLIDATED_NUMBER, L2_RISE_CONSOLIDATED_NAME, CUST_ACCOUNT_TYPE, CUSTOMER_SEGMENT, CUSTOMER_SEGMENT_LEVEL, CHURNED_FLAG, CHURN_DATE FROM RUS_AIML.PS_DOCUWARE_L1_CUST;

# CREATE OR REPLACE VIEW RUS_AIML.DNB_RISK_BREAKDOWN_V AS SELECT TO_CHAR(ACCOUNT_NUMBER) AS ACCOUNT_NUMBER, OVERALL_BUSINESS_RISK, RICOH_CUSTOM_RISK_MODEL, PROBABILITY_OF_DELINQUENCY, PAYMENT_RISK_TRIPLE_A_RATING FROM RUS_AIML.DNB_RISK_BREAKDOWN;

# CREATE OR REPLACE VIEW RUS_AIML.DOCUWARE_USAGE_JAPAN_V1_LATEST_V AS
# --select * from RUS_AIML.DOCUWARE_USAGE_JAPAN_V1_LATEST_V_IMPUTED; -- where YYYYWK > 202152;
#  WITH latback AS (
#    SELECT DISTINCT CONTRACT_NUMBER
#     ,REGEXP_REPLACE(trim(CUSTOMER_NAME), '  ', ' ') AS CUSTOMER_NAME,
#     CONTRACT_START
#     ,CONTRACT_END
#     ,DOCUMENTS_OPENED
#     ,USED_STORAGE__MB
#     ,CONTRACT_LINE_ITEMS
#     ,PERIOD
#     ,YYYYWK
#    FROM PS_DOCUWARE_USAGE_COMBINED --DOCUWARE_USAGE_JAPAN_20250513_SNAP
#    WHERE RICOH_REGIONS = 'US/Canada'
#     AND ADP__CURRENT != 'Ricoh Canada Inc.'
#    )
#   ,jaro AS (
#    SELECT DISTINCT a.CUST_ACCOUNT_NUMBER,
#    a.churn_date,
#    a.churned_flag
#     ,b.CONTRACT_NUMBER
#     ,a.CUST_PARTY_NAME
#     ,b.CUSTOMER_NAME,
#     b.CONTRACT_START
#     ,b.CONTRACT_END
#     ,b.DOCUMENTS_OPENED
#     ,b.USED_STORAGE__MB
#     ,b.CONTRACT_LINE_ITEMS
#     ,b.PERIOD
#     ,b.YYYYWK
#     ,jarowinkler_similarity(a.CUST_PARTY_NAME, b.CUSTOMER_NAME)
#     ,rank() OVER (
#      PARTITION BY a.CUST_ACCOUNT_NUMBER ORDER BY jarowinkler_similarity(a.CUST_PARTY_NAME, b.CUSTOMER_NAME) DESC
#      ) AS match_rank
#    FROM RUS_AIML.PS_DOCUWARE_L1_CUST a
#    FULL OUTER JOIN latback b
#    WHERE jarowinkler_similarity(a.CUST_PARTY_NAME, b.CUSTOMER_NAME) BETWEEN 93
#      AND 100
#    )
# SELECT *
# FROM jaro
# WHERE jaro.match_rank = 1
#  AND EXISTS (
#   SELECT 1
#   FROM jaro
#   );
# %%
# class DocuwareDataProcessor:
#     def __init__(self):
#         """
#         Initialize Snowflake session with your connection parameters
#         """
#         self.session = Session.builder.configs(connection_parameters).create()
#         self.session.use_schema("RUS_AIML")
#         self.session.use_role("RAC_PROD_AIML_ROLE")

#     def get_payments_view(self):
#         """
#         Equivalent to: CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_PAYMENTS_V
#         """
#         return self.session.table("RUS_AIML.PS_DOCUWARE_PAYMENTS").select(
#             col("CUSTOMER_NO"),
#             col("RECEIPT_DATE"),
#             col("FUNCTIONAL_AMOUNT")
#         )

#     def get_revenue_view(self):
#         """
#         Equivalent to: CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_REVENUE_V
#         """
#         return self.session.table("RUS_AIML.PS_DOCUWARE_REVENUE").select(
#             col("CUST_ACCOUNT_NUMBER"),
#             col("DATE_INVOICE_GL_DATE"),
#             col("INVOICE_REVLINE_TOTAL")
#         )

#     def get_trx_view(self):
#         """
#         Equivalent to: CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_TRX_V
#         """
#         return self.session.table("RUS_AIML.PS_DOCUWARE_TRX").select(
#             col("ACCOUNT_NUMBER"),
#             col("TRX_DATE"),
#             col("ORIGINAL_AMOUNT_DUE")
#         )

#     def get_contracts_subline_view(self):
#         """
#         Equivalent to: CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_CONTRACTS_SUBLINE_V
#         """
#         return self.session.table("RUS_AIML.PS_DOCUWARE_CONTRACT_SUBLINE").select(
#             col("CUST_ACCOUNT_NUMBER"),
#             col("SLINE_START_DATE"),
#             col("SLINE_END_DATE"),
#             col("SLINE_STATUS")
#         )

#     def get_renewals_view(self):
#         """
#         Equivalent to: CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_RENEWALS_V
#         """
#         return self.session.table("RUS_AIML.PS_DOCUWARE_SSCD_RENEWALS").select(
#             to_char(col("BILLTOCUSTOMERNUMBER")).alias("BILLTOCUSTOMERNUMBER"),
#             to_char(col("SHIPTOCUSTNUM")).alias("SHIPTOCUSTNUM"),
#             col("STARTDATECOVERAGE"),
#             col("CONTRACT_END_DATE")
#         )

#     def get_l1_cust_view(self):
#         """
#         Equivalent to: CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_L1_CUST_V
#         """
#         return self.session.table("RUS_AIML.PS_DOCUWARE_L1_CUST").select(
#             col("CUST_ACCOUNT_NUMBER"),
#             col("CUST_PARTY_NAME"),
#             col("L3_RISE_CONSOLIDATED_NUMBER"),
#             col("L3_RISE_CONSOLIDATED_NAME"),
#             col("L2_RISE_CONSOLIDATED_NUMBER"),
#             col("L2_RISE_CONSOLIDATED_NAME"),
#             col("CUST_ACCOUNT_TYPE"),
#             col("CUSTOMER_SEGMENT"),
#             col("CUSTOMER_SEGMENT_LEVEL"),
#             col("CHURNED_FLAG"),
#             col("CHURN_DATE")
#         )

#     def get_dnb_risk_breakdown_view(self):
#         """
#         Equivalent to: CREATE OR REPLACE VIEW RUS_AIML.DNB_RISK_BREAKDOWN_V
#         """
#         return self.session.table("RUS_AIML.DNB_RISK_BREAKDOWN").select(
#             to_char(col("ACCOUNT_NUMBER")).alias("ACCOUNT_NUMBER"),
#             col("OVERALL_BUSINESS_RISK"),
#             col("RICOH_CUSTOM_RISK_MODEL"),
#             col("PROBABILITY_OF_DELINQUENCY"),
#             col("PAYMENT_RISK_TRIPLE_A_RATING")
#         )

#     def get_docuware_usage_japan_latest_view(self):
#         """
#         Equivalent to: CREATE OR REPLACE VIEW RUS_AIML.DOCUWARE_USAGE_JAPAN_V1_LATEST_V
#         Complex view with CTEs, joins, and Jaro-Winkler similarity
#         """
#         # Step 1: Create latback CTE equivalent
#         latback = (self.session.table("PS_DOCUWARE_USAGE_COMBINED")
#                   .filter((col("RICOH_REGIONS") == "US/Canada") &
#                          (col("ADP__CURRENT") != "Ricoh Canada Inc."))
#                   .select(
#                       col("CONTRACT_NUMBER"),
#                       regexp_replace(trim(col("CUSTOMER_NAME")), F.lit("  "), F.lit(" ")).alias("CUSTOMER_NAME"),
#                       col("CONTRACT_START"),
#                       col("CONTRACT_END"),
#                       col("DOCUMENTS_OPENED"),
#                       col("USED_STORAGE__MB"),
#                       col("CONTRACT_LINE_ITEMS"),
#                       col("PERIOD"),
#                       col("YYYYWK")
#                   )
#                   .distinct())

#         # Step 2: Get L1_CUST data
#         l1_cust = self.get_l1_cust_view()

#         # Step 3: Join with Jaro-Winkler similarity
#         # Note: Snowflake has JAROWINKLER_SIMILARITY function
#         jaro = (l1_cust.alias("a")
#                .join(latback.alias("b"),
#                     F.call_function("JAROWINKLER_SIMILARITY",
#                                    col("a.CUST_PARTY_NAME"),
#                                    col("b.CUSTOMER_NAME")).between(93, 100),
#                     "full_outer")
#                .select(
#                    col("a.CUST_ACCOUNT_NUMBER"),
#                    col("a.churn_date"),
#                    col("a.churned_flag"),
#                    col("b.CONTRACT_NUMBER"),
#                    col("a.CUST_PARTY_NAME"),
#                    col("b.CUSTOMER_NAME"),
#                    col("b.CONTRACT_START"),
#                    col("b.CONTRACT_END"),
#                    col("b.DOCUMENTS_OPENED"),
#                    col("b.USED_STORAGE__MB"),
#                    col("b.CONTRACT_LINE_ITEMS"),
#                    col("b.PERIOD"),
#                    col("b.YYYYWK"),
#                    F.call_function("JAROWINKLER_SIMILARITY",
#                                   col("a.CUST_PARTY_NAME"),
#                                   col("b.CUSTOMER_NAME")).alias("jarowinkler_similarity"),
#                    F.rank().over(
#                        Window.partition_by(col("a.CUST_ACCOUNT_NUMBER"))
#                        .order_by(F.call_function("JAROWINKLER_SIMILARITY",
#                                                col("a.CUST_PARTY_NAME"),
#                                                col("b.CUSTOMER_NAME")).desc())
#                    ).alias("match_rank")
#                ))

#         # Step 4: Filter to rank 1 matches only
#         result = jaro.filter(col("match_rank") == 1)

#         return result

#     def create_actual_views_in_snowflake(self):
#         """
#         Create the actual views in Snowflake database using the original SQL
#         """
#         view_sqls = [
#             """CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_PAYMENTS_V AS
#                SELECT CUSTOMER_NO, RECEIPT_DATE, FUNCTIONAL_AMOUNT
#                FROM RUS_AIML.PS_DOCUWARE_PAYMENTS""",

#             """CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_REVENUE_V AS
#                SELECT CUST_ACCOUNT_NUMBER, DATE_INVOICE_GL_DATE, INVOICE_REVLINE_TOTAL
#                FROM RUS_AIML.PS_DOCUWARE_REVENUE""",

#             """CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_TRX_V AS
#                SELECT ACCOUNT_NUMBER, TRX_DATE, ORIGINAL_AMOUNT_DUE
#                FROM RUS_AIML.PS_DOCUWARE_TRX""",

#             """CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_CONTRACTS_SUBLINE_V AS
#                SELECT CUST_ACCOUNT_NUMBER, SLINE_START_DATE, SLINE_END_DATE, SLINE_STATUS
#                FROM RUS_AIML.PS_DOCUWARE_CONTRACT_SUBLINE""",

#             """CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_RENEWALS_V AS
#                SELECT TO_CHAR(BILLTOCUSTOMERNUMBER) AS BILLTOCUSTOMERNUMBER,
#                       TO_CHAR(SHIPTOCUSTNUM) AS SHIPTOCUSTNUM,
#                       STARTDATECOVERAGE, CONTRACT_END_DATE
#                FROM RUS_AIML.PS_DOCUWARE_SSCD_RENEWALS""",

#             """CREATE OR REPLACE VIEW RUS_AIML.PS_DOCUWARE_L1_CUST_V AS
#                SELECT CUST_ACCOUNT_NUMBER, CUST_PARTY_NAME, L3_RISE_CONSOLIDATED_NUMBER,
#                       L3_RISE_CONSOLIDATED_NAME, L2_RISE_CONSOLIDATED_NUMBER,
#                       L2_RISE_CONSOLIDATED_NAME, CUST_ACCOUNT_TYPE, CUSTOMER_SEGMENT,
#                       CUSTOMER_SEGMENT_LEVEL, CHURNED_FLAG, CHURN_DATE
#                FROM RUS_AIML.PS_DOCUWARE_L1_CUST""",

#             """CREATE OR REPLACE VIEW RUS_AIML.DNB_RISK_BREAKDOWN_V AS
#                SELECT TO_CHAR(ACCOUNT_NUMBER) AS ACCOUNT_NUMBER, OVERALL_BUSINESS_RISK,
#                       RICOH_CUSTOM_RISK_MODEL, PROBABILITY_OF_DELINQUENCY,
#                       PAYMENT_RISK_TRIPLE_A_RATING
#                FROM RUS_AIML.DNB_RISK_BREAKDOWN"""
#         ]

#         for sql in view_sqls:
#             self.session.sql(sql).collect()
#             print(f"Created view: {sql.split('VIEW')[1].split('AS')[0].strip()}")
# # %%
# processor = DocuwareDataProcessor()

# payments_df = processor.get_payments_view().to_pandas()

# print(payments_df.head().to_string())

# revenue_df = processor.get_revenue_view().to_pandas()
# print(revenue_df.head().to_string())

# %%
# def create_docuware_usage_japan_v1_latest_v(session: Session) -> pd.DataFrame:
#     """Equivalent of RUS_AIML.DOCUWARE_USAGE_JAPAN_V1_LATEST_V view"""

#     # Step 2: Create jaro CTE equivalent
#     jaro_query = """
#         WITH latback AS (
#             SELECT DISTINCT CONTRACT_NUMBER,
#                    REGEXP_REPLACE(trim(CUSTOMER_NAME), '  ', ' ') AS CUSTOMER_NAME,
#                    CONTRACT_START,
#                    CONTRACT_END,
#                    DOCUMENTS_OPENED,
#                    USED_STORAGE__MB,
#                    CONTRACT_LINE_ITEMS,
#                    PERIOD,
#                    YYYYWK
#             FROM RUS_AIML.DOCUWARE_USAGE_COMBINED_V
#         ),
#         jaro AS (
#             SELECT DISTINCT a.CUST_ACCOUNT_NUMBER,
#                    a.churn_date,
#                    a.churned_flag,
#                    b.CONTRACT_NUMBER,
#                    a.CUST_PARTY_NAME,
#                    b.CUSTOMER_NAME,
#                    b.CONTRACT_START,
#                    b.CONTRACT_END,
#                    b.DOCUMENTS_OPENED,
#                    b.USED_STORAGE__MB,
#                    b.CONTRACT_LINE_ITEMS,
#                    b.PERIOD,
#                    b.YYYYWK,
#                    jarowinkler_similarity(a.CUST_PARTY_NAME, b.CUSTOMER_NAME),
#                    rank() OVER (
#                        PARTITION BY a.CUST_ACCOUNT_NUMBER
#                        ORDER BY jarowinkler_similarity(a.CUST_PARTY_NAME, b.CUSTOMER_NAME) DESC
#                    ) AS match_rank
#             FROM RUS_AIML.PS_DOCUWARE_CUST_FINAL a
#             FULL OUTER JOIN latback b
#             WHERE jarowinkler_similarity(a.CUST_PARTY_NAME, b.CUSTOMER_NAME) BETWEEN 93 AND 100
#         )
#         SELECT *
#         FROM jaro
#         WHERE jaro.match_rank = 1
#     """

#     return session.sql(jaro_query).to_pandas()
# %%
# trx_df = processor.get_trx_view().to_pandas()
# contracts_subline_df = processor.get_contracts_subline_view().to_pandas()
# renewals_df = processor.get_renewals_view().to_pandas()
# l1_cust_df = processor.get_l1_cust_view().to_pandas()
# dnb_risk_df = processor.get_dnb_risk_breakdown_view().to_pandas()

# dnb_risk_df.head()

# usage_japan_df = processor.get_docuware_usage_japan_latest_view().to_pandas()
# %%
# Justin Bovard 2025-07-14
# Impute_Missing_Usage_23
# Code to impute missing usage data for first half of 2023
#Code to impute missing usage data for first half of 2023
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
# Focus only on Contract Start Date >= ‘2020-01-01’
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
imputations = imputations.drop(columns=['ywk']) #'Unnamed: 0' in snapshot
usage = pd.concat([usageFIX,imputations])
usage_latest = usage
# %%
usage['YYYYWK']
# %%
# Fetching from previous SPROC output table
#  cell9
raw_df = session.sql("SELECT * FROM RAC_RAPID_DEV.RUS_AIML.PS_DOCUWARE_RAW_DATA_EXTRACTION").to_pandas()

non_ts_numeric_cols = ["PROBABILITY_OF_DELINQUENCY", "RICOH_CUSTOM_RISK_MODEL"]
non_ts_categorical_cols = ["OVERALL_BUSINESS_RISK", "PAYMENT_RISK_TRIPLE_A_RATING", "CONTRACT_LINE_ITEMS"]
columns_to_be_processed_later = non_ts_numeric_cols + non_ts_categorical_cols + ["CUST_ACCOUNT_NUMBER", "LIFESPAN_MONTHS"]
finalized_df_ohe_to_process = raw_df.groupby("CUST_ACCOUNT_NUMBER")[columns_to_be_processed_later].first()

# Imputation for Non Time Series columns
pofd_median = finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"].median()
finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"] = finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"].apply(lambda x:   float(pofd_median) if np.isnan(x) else x)

temp=finalized_df_ohe_to_process["RICOH_CUSTOM_RISK_MODEL"].mode().to_frame()
rcrm_mode = temp.loc[temp.index[0], 'RICOH_CUSTOM_RISK_MODEL']
finalized_df_ohe_to_process["RICOH_CUSTOM_RISK_MODEL"] = finalized_df_ohe_to_process["RICOH_CUSTOM_RISK_MODEL"].apply(lambda x: float(rcrm_mode) if np.isnan(x) else x)
# %% [markdown]
# # Preprocessing Raw Data
# %%
# Sproc_2
# Fetching all the different views
# Justin Bovard 2025-07-14
#Fetching all the different views
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
# %%
usage_latest["YYYYWK_Transformed"] = pd.to_datetime(usage_latest["YYYYWK"].apply(convert_yyyywk_to_actual_mid_date), errors = "coerce")
# %%
usage_latest["YYYYWK_Transformed"]
# %%
usage_latest_churned = usage_latest.merge(l1_cust_churned[["CUST_ACCOUNT_NUMBER"]], on="CUST_ACCOUNT_NUMBER",  how="inner")

# Merging all the churned customers data frames i.e. Payments, Revenue, Transactions, contracts, contracts subline, contracts topline, renewals, snow inc, tech survey, loyalty survey, dnb risk and usage latest

merged_1 = p_r_t_merged.merge(contracts_sub_df_churned, left_on = ["CUST_ACCOUNT_NUMBER", "MONTH"], right_on = ["CUST_ACCOUNT_NUMBER", "SLINE_START_DATE"], how="outer")
merged_1["MONTH"] = merged_1["MONTH"].fillna(merged_1["SLINE_START_DATE"])
merged_1.head()

merged_2 = merged_1.merge(renewals_df_churned_2, left_on = ["CUST_ACCOUNT_NUMBER","MONTH"], right_on = ["CUST_ACCOUNT_NUMBER","STARTDATECOVERAGE"], how="outer")
merged_2["MONTH"] = merged_2["MONTH"].fillna(merged_2["STARTDATECOVERAGE"])
# %%
usage_latest_churned["YYYYWK_MONTH"].unique()
# %%
merged_2["STARTDATECOVERAGE"].unique()
# %%
merged_3 = merged_2.merge(dnb_risk_df_churned, on="CUST_ACCOUNT_NUMBER", how="left")

merged_4 = merged_3.merge(usage_latest_churned, left_on= ["CUST_ACCOUNT_NUMBER","MONTH"], right_on = ["CUST_ACCOUNT_NUMBER", "YYYYWK_MONTH"], how="outer")
merged_4["MONTH"] = merged_4["MONTH"].fillna(merged_4["YYYYWK_MONTH"])

to_drop_2 = ["CONTRACT_NUMBER", "CUST_PARTY_NAME", "CUSTOMER_NAME", "CONTRACT_END","JAROWINKLER_SIMILARITY(A.CUST_PARTY_NAME, B.CUSTOMER_NAME)"]
#(A.CUST_PARTY_NAME, B.CUSTOMER_NAME) removed from jaro winkler

merged_5 = merged_4.drop(to_drop_2, axis=1)

merged_5 = merged_4.merge(l1_cust_churned[["CUST_ACCOUNT_NUMBER", "CHURNED_FLAG", "CHURN_DATE"]], on = "CUST_ACCOUNT_NUMBER", how="inner")

merged_5["EARLIEST_DATE"] = merged_5[["RECEIPT_DATE", "DATE_INVOICE_GL_DATE", "TRX_DATE", "SLINE_START_DATE", "STARTDATECOVERAGE"]].min(axis=1)
merged_5["FINAL_EARLIEST_DATE"] = merged_5.groupby("CUST_ACCOUNT_NUMBER")["EARLIEST_DATE"].transform("min")

merged_5["CHURN_DATE"] = pd.to_datetime(merged_5["CHURN_DATE"])
merged_5["CHURN_MONTH"] = pd.to_datetime(merged_5["CHURN_DATE"]).dt.to_period("M").dt.to_timestamp()


merged_5["LIFESPAN_MONTHS"] = ((merged_5["CHURN_DATE"] - merged_5["FINAL_EARLIEST_DATE"]).dt.days) / 30
merged_5["DAYS_TO_CHURN"]  = ((merged_5["CHURN_DATE"] - merged_5["FINAL_EARLIEST_DATE"]).dt.days)

to_drop_3 = ["SLINE_END_DATE", "SLINE_STATUS", "SUB_EARLIEST_DATE", "SUB_LATEST_DATE", "RENEWALS_EARLIEST_DATE", "RENEWALS_LATEST_DATE", "CONTRACT_END_DATE", "CHURNED_FLAG","CHURN_MONTH", "EARLIEST_DATE"]

merged_5 = merged_5.drop(to_drop_3, axis=1)
# %%
merged_3[['CUST_ACCOUNT_NUMBER', 'MONTH']]
# %%
usage_latest_churned[['CUST_ACCOUNT_NUMBER', 'YYYYWK_MONTH']]
# %%
merged_3['CUST_ACCOUNT_NUMBER'].nunique()
# %%
usage_latest_churned['CUST_ACCOUNT_NUMBER'].nunique()
# %%
merged_4['YYYYWK']
# %%
merged_5.columns
# %%
merged_5["STARTDATECOVERAGE"].unique()
# %%
merged_5["YYYYWK_MONTH"].unique()
# %%
merged_5["DATE_INVOICE_GL_DATE"].unique()
# %%
date_col = ["MONTH","YYYYWK_Transformed", "FINAL_EARLIEST_DATE", "CHURN_DATE"]
merged_5[date_col] = merged_5[date_col].astype(str)
# %%
merged_5 = merged_5.drop_duplicates()
# %%
merged_5.head()
# %%
#usage_latest_churned['CUST_ACCOUNT_NUMBER'].unique()
temp = pd.DataFrame({'CUST_ACCOUNT_NUMBER':usage_latest_churned['CUST_ACCOUNT_NUMBER'].unique()})
merge_final = merged_5.merge(temp, on='CUST_ACCOUNT_NUMBER', how='inner')
# %%
merge_final["DATE_INVOICE_GL_DATE"].unique()
# %%
merge_final["STARTDATECOVERAGE"].unique()
# %%
merge_final["YYYYWK_MONTH"].unique()
# %%
merge_final["RECEIPT_DATE"].unique()
# %%
merge_final["RECEIPT_DATE"]= pd.to_datetime(merge_final["RECEIPT_DATE"])
merge_final["YYYYWK_MONTH"]= pd.to_datetime(merge_final["YYYYWK_MONTH"])
merge_final["DATE_INVOICE_GL_DATE"] = pd.to_datetime(merge_final["DATE_INVOICE_GL_DATE"])

# %%
merge_final["DATE_INVOICE_GL_DATE"].head()
# %%
def timestamp_To_Date(row):
    if type(row) == pd._libs.tslibs.nattype.NaTType:
        return None
    else:
        return pd.Timestamp(row).strftime('%Y-%m-%d')

# %%
merge_final["DATE_INVOICE_GL_DATE_CANONICAL"] = merge_final["DATE_INVOICE_GL_DATE"].apply(timestamp_To_Date)
merge_final["DATE_INVOICE_GL_DATE_CANONICAL"].head()
# %%
merge_final["RECEIPT_DATE"].head()
# %%
merge_final["RECEIPT_DATE_CANONICAL"] = merge_final["RECEIPT_DATE"].apply(timestamp_To_Date)
merge_final["RECEIPT_DATE_CANONICAL"].head()
# %%
merge_final["YYYYWK_MONTH_CANONICAL"] = merge_final["YYYYWK_MONTH"].apply(timestamp_To_Date)
merge_final["YYYYWK_MONTH_CANONICAL"].head(100)
# %%
merge_final.drop(["DATE_INVOICE_GL_DATE", "RECEIPT_DATE", "YYYYWK_MONTH"], inplace=True, axis=1)
merge_final.rename(columns={"DATE_INVOICE_GL_DATE_CANONICAL":"DATE_INVOICE_GL_DATE", "RECEIPT_DATE_CANONICAL":"RECEIPT_DATE", "YYYYWK_MONTH_CANONICAL":"YYYYWK_MONTH"}, inplace=True)
# %%
merge_final.head()
# %%
#CK added to convert col from str to int
merge_final['YYYYWK'] = np.floor(pd.to_numeric(merge_final['YYYYWK'], errors='coerce')).astype('Int64')
# %%
session.write_pandas(merge_final, "PS_DOCUWARE_RAW_DATA_EXTRACTION", auto_create_table=True, overwrite = True)
# %%
# CK removed DB from query RAC_RAPID_DEV
raw_df = session.sql("SELECT * FROM RUS_AIML.PS_DOCUWARE_RAW_DATA_EXTRACTION").to_pandas()
# %%
raw_df.head()
# %%
raw_df
# %% [markdown]
# # Data Imputation and Feature Engineering to make Training Set
# %%
#try:
#Fetching from previous SPROC output table
raw_df = session.sql("SELECT * FROM RUS_AIML.PS_DOCUWARE_RAW_DATA_EXTRACTION").to_pandas()

non_ts_numeric_cols = ["PROBABILITY_OF_DELINQUENCY", "RICOH_CUSTOM_RISK_MODEL"]
non_ts_categorical_cols = ["OVERALL_BUSINESS_RISK", "PAYMENT_RISK_TRIPLE_A_RATING", "CONTRACT_LINE_ITEMS"]
columns_to_be_processed_later = non_ts_numeric_cols + non_ts_categorical_cols + ["CUST_ACCOUNT_NUMBER", "LIFESPAN_MONTHS", "DAYS_TO_CHURN"]
finalized_df_ohe_to_process = raw_df.groupby("CUST_ACCOUNT_NUMBER")[columns_to_be_processed_later].first()

# Imputation for Non Time Series columns
pofd_median = finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"].median()
finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"] = finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"].apply(lambda x:   float(pofd_median) if np.isnan(x) else x)

#rcrm_mode = finalized_df_ohe_to_process["RICOH_CUSTOM_RISK_MODEL"].mode()
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

now = datetime.now()
date_string = now.strftime("%m-%d-%Y")

#encoder_filename = "@RAC_RAPID_DEV.RUS_AIML.PS_DOCUWARE_CHURN/ps_docuware_target_encoder_object/ENC_CURRENT.joblib"
#encoder_filename_ar = "@RAC_RAPID_DEV.RUS_AIML.PS_DOCUWARE_CHURN/ps_docuware_target_encoder_object/archive/ENC_"+date_string+".joblib"

#joblib.dump(enc,  encoder_filename)
#joblib.dump(enc, encoder_filename_ar)

import_dir = sys._xoptions.get("snowflake_import_directory")
model_file = os.path.join("/tmp", "ENC_CURRENT.joblib.gz")
dump(enc, model_file)
#session.file.put(model_file, "@PS_DOCUWARE_CHURN",overwrite=True)
session.file.put(model_file, "@PS_DOCUWARE_CHURN/ps_docuware_target_encoder_object",overwrite=True)

model_file_ar = os.path.join("/tmp", "ENC_"+date_string+".joblib.gz")
dump(enc, model_file_ar)
session.file.put(model_file_ar, "@PS_DOCUWARE_CHURN/ps_docuware_target_encoder_object/archive",overwrite=True)

# Imputation for Time Series columns
ts_columns = ["CUST_ACCOUNT_NUMBER", "YYYYWK", "DOCUMENTS_OPENED", "USED_STORAGE__MB", "INVOICE_REVLINE_TOTAL", "ORIGINAL_AMOUNT_DUE", "FUNCTIONAL_AMOUNT"]
raw_df["transformed_YYYYWK"] = raw_df["MONTH"].apply(convert_date_to_yyyywk)
#Impute missing YYYYWK with equivalent MONTH
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
# %%
raw_df
# %%
only_features
# %%
only_features.head()
# %%
lifespan_df = raw_df[['CUST_ACCOUNT_NUMBER', 'DAYS_TO_CHURN']]
lifespan_df['CUST_ACCOUNT_NUMBER'] = lifespan_df['CUST_ACCOUNT_NUMBER'].astype(int)
lifespan_df.drop_duplicates(inplace=True)
lifespan_df.shape
# %%
df_churn_cust = l1_cust_churned[["CUST_ACCOUNT_NUMBER", "CHURN_DATE"]]
df_churn_cust['CHURN_YYYYWK_DATE'] = df_churn_cust['CHURN_DATE'].apply(convert_date_to_yyyywk)
df_churn_cust['CUST_ACCOUNT_NUMBER'] = df_churn_cust['CUST_ACCOUNT_NUMBER'].astype(int)
#print(only_features.shape) # (14614, 7)
df_raw_churn = only_features.merge(df_churn_cust, on='CUST_ACCOUNT_NUMBER', how='inner')
#print(df_raw_churn.shape) # (14614, 9)
final_features = df_raw_churn[df_raw_churn['YYYYWK'] <= df_raw_churn['CHURN_YYYYWK_DATE']]
print(final_features.shape)
# %%
final_features.head()
# %%
final_features_for_FE = final_features.drop(['CHURN_DATE', 'CHURN_YYYYWK_DATE'], axis=1)
#print(final_features_for_FE.shape) # (5551, 7)
final_features_for_FE = final_features_for_FE.merge(lifespan_df, on='CUST_ACCOUNT_NUMBER', how='inner')
#print(final_features_for_FE.shape) # (5551, 8)
final_features_for_FE.head()
# %%
final_features_for_FE.shape
# %%
final_features_for_FE.head()
# %%
print(X_train_encoded['CONTRACT_LINE_ITEMS'])
# %% [markdown]
# # Store time series and Non time series data into DB before feature engineering from TSFRESH
# %%
for col in X_train_encoded.columns:
    if pd.api.types.is_sparse(X_train_encoded[col]):
        X_train_encoded[col] = X_train_encoded[col].sparse.to_dense()
# %%
# WITH fi(CUST_ACCOUNT_NUMBER, YYYYWK) AS(
#     SELECT CUST_ACCOUNT_NUMBER, min(YYYYWK) AS YY FROM RUS_AIML.DOCUWARE_USAGE_JAPAN_V1_LATEST_V GROUP BY(CUST_ACCOUNT_NUMBER)
# )

# SELECT a.CUST_ACCOUNT_NUMBER, a.YYYYWK FROM fi a join RUS_AIML.PS_DOCUWARE_L1_CUST b on a.CUST_ACCOUNT_NUMBER = b.CUST_ACCOUNT_NUMBER where b.CHURNED_FLAG=True;

# %%
# all_usage = cell177.to_pandas()
# Direct SQL approach - replace your SQL with this:
query = """
WITH fi(CUST_ACCOUNT_NUMBER, YYYYWK) AS(
    SELECT CUST_ACCOUNT_NUMBER, min(YYYYWK) AS YY FROM RUS_AIML.DOCUWARE_USAGE_JAPAN_V1_LATEST_V GROUP BY(CUST_ACCOUNT_NUMBER)
)

SELECT a.CUST_ACCOUNT_NUMBER, a.YYYYWK FROM fi a join RUS_AIML.PS_DOCUWARE_L1_CUST b on a.CUST_ACCOUNT_NUMBER = b.CUST_ACCOUNT_NUMBER where b.CHURNED_FLAG='Y';
"""
# %%
all_usage = session.sql(query).to_pandas()
# %%
raw_df = session.sql("SELECT * FROM RAC_RAPID_DEV.RUS_AIML.PS_DOCUWARE_RAW_DATA_EXTRACTION").to_pandas()
df_cust_earliest_date = raw_df[['CUST_ACCOUNT_NUMBER', 'FINAL_EARLIEST_DATE']]
df_cust_earliest_date.drop_duplicates(inplace=True)
df_cust_earliest_date['CUST_ACCOUNT_NUMBER'] = df_cust_earliest_date['CUST_ACCOUNT_NUMBER'].astype(int)
all_usage['CUST_ACCOUNT_NUMBER'] = all_usage['CUST_ACCOUNT_NUMBER'].astype(int)
df_merged = pd.merge(df_cust_earliest_date, all_usage , on='CUST_ACCOUNT_NUMBER', how='inner')
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
final_features_for_FE_trimmed
# %%
final_features_for_FE_trimmed.head()
# %%
session.write_pandas(X_train_encoded, "PS_DOCUWARE_RAW_X_TRAIN_ENCODED_DATA_BEFORE_TSFRESH", auto_create_table=True, overwrite = True)
session.write_pandas(final_features_for_FE_trimmed, "PS_DOCUWARE_RAW_DATA_BEFORE_TSFRESH", auto_create_table=True, overwrite = True)
# %% [markdown]
# # Starting TSFRESH API
# %%
#ts_comprehensive_df = time_series_ts_fresh_features(only_features)
with timer():
    ts_comprehensive_df, final_ts_age  = engineer_timeseries_cols_using_tsfresh(final_features_for_FE_trimmed)
# %%
#ts_comprehensive_df = session.sql("SELECT * FROM RAC_RAPID_DEV.RUS_AIML.PS_DOCUWARE_ONLY_TS_DF").to_pandas()
#except Exception as e:
#    print("FAIL(Post Processing)!" + " Error: " + str(e))
ts_comprehensive_df = ts_comprehensive_df.reset_index(drop=True)
final_ts_age = final_ts_age.reset_index(drop=True)

final_ts_df = pd.concat([ts_comprehensive_df, final_ts_age], axis=1, ignore_index=True)
#session.write_pandas(final_ts_df, "PS_DOCUWARE_TSFRESH_FEATURES_WITH_AGE", auto_create_table=True, overwrite = True)

all_columns = ts_comprehensive_df.columns.tolist()+final_ts_age.columns.tolist()
final_ts_df.columns = all_columns
final_ts_df.head()
# %%
final_ts_age
# %%
# %%
ts_comprehensive_df = final_ts_df.copy()
ts_comprehensive_df.shape
# %%
# Check if it contains CUST_ACCOUNT_NUMBER related TSFRESH features
for col in ts_comprehensive_df.columns:
    if re.search("^CUST_ACCOUNT_NUMBER_", col):
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
# %%
ts_comprehensive_df["CUST_ACCOUNT_NUMBER"] = ts_comprehensive_df["CUST_ACCOUNT_NUMBER"].astype(int)
X_train_encoded["CUST_ACCOUNT_NUMBER"] = X_train_encoded["CUST_ACCOUNT_NUMBER"].astype(int)
X_train_encoded = X_train_encoded.reset_index(drop=True)
ts_comprehensive_df = ts_comprehensive_df.reset_index(drop=True)
comprehensive_imputed_df = pd.merge(ts_comprehensive_df, X_train_encoded, on="CUST_ACCOUNT_NUMBER", how="inner")
# %%
ts_comprehensive_df.shape, X_train_encoded.shape, comprehensive_imputed_df.shape
# %%
for col in ts_comprehensive_df.columns:
    if re.search("^DAYS", col):
        print(col)
# %%
for col in comprehensive_imputed_df.columns:
    if re.search("^DAYS", col):
        print(col)
# %%
with timer():
    for col in comprehensive_imputed_df.columns:
        if pd.api.types.is_sparse(comprehensive_imputed_df[col]):
            comprehensive_imputed_df[col] = comprehensive_imputed_df[col].sparse.to_dense()

    session.write_pandas(comprehensive_imputed_df, "PS_DOCUWARE_IMPUTED_DATA", auto_create_table=True, overwrite = True)
    #session.write_pandas(comprehensive_imputed_df, "PS_DOCUWARE_IMPUTED_DATA_266", auto_create_table=True, overwrite = True)
    print("Successfuly created table PS_DOCUWARE_IMPUTED_DATA")
# %%
ts_comprehensive_df.shape
# %%
comprehensive_imputed_df.shape
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
prefix
# %% [markdown]
# #### Training using Catboost Regression
# %%
comprehensive_imputed_df = session.sql("SELECT * FROM RUS_AIML.PS_DOCUWARE_IMPUTED_DATA").to_pandas()
comprehensive_imputed_df = comprehensive_imputed_df.replace([np.inf, -np.inf], 0)
comprehensive_imputed_df = comprehensive_imputed_df.fillna(0)

with timer():
    for col in comprehensive_imputed_df.columns:
        comprehensive_imputed_df[col] = pd.to_numeric(comprehensive_imputed_df[col], errors="coerce").astype(float)

# comprehensive_imputed_o_df = pd.series(comprehensive_imputed_df)
# comprehensive_imputed_df = pd.to_numeric(comprehensive_imputed_o_df, errors="coerce")
# comprehensive_imputed_df = comprehensive_imputed_df.to_frame()
# %%
cust_churn_df = comprehensive_imputed_df[["CUST_ACCOUNT_NUMBER","DAYS_TO_CHURN"]]
cust_churn_df.head()
# %%
all_columns = comprehensive_imputed_df.columns
for cols in all_columns:
    if re.search("^CUST_ACCOUNT_NUMBER" , cols):
        print(cols)
# %%
cust_churn_df.drop_duplicates(inplace=True)
cust_churn_df.shape
# %%
cust_churn_df.head()
# %% [markdown]
# #### Stratified Sampling
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
stratified_df.head()
# %%
stratified_df = stratified_df[stratified_df['SAMPLE']!=5]
features = stratified_df.drop(["DAYS_TO_CHURN", "SAMPLE"], axis=1)
y = stratified_df["SAMPLE"]
# %%
features
# %%
y
# %%
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
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
comprehensive_imputed_df.shape
# %%
comprehensive_imputed_df
# %%
X_train_all_cols = pd.merge(comprehensive_imputed_df, X_train, how='inner', on='CUST_ACCOUNT_NUMBER')
X_train_all_cols.shape
# %%
X_test_all_cols = pd.merge(comprehensive_imputed_df, X_test, how='inner', on='CUST_ACCOUNT_NUMBER')
X_test_all_cols.shape
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

param_grid = {"learning_rate": Real(0.2, .4, "uniform"), #was (.10,1)
             "max_depth": Integer(9, 12),  #(2,12)
             "subsample": Real(0.1, .2, "uniform"), # was (0.1, 1.0,
             "colsample_bytree": Real(0.5, 1.0, "uniform"), # subsample ratio of columns by tree min was .1
             "reg_lambda": Real(20., 40., "uniform"), # L2 regularization # was (1e-9, 100., "uniform")
             "reg_alpha": Real(5., 20., "uniform"), # L1 regularization # was (1e-9, 100., "uniform")
             "n_estimators": Integer(500, 1000) #was (20,3000)
}

# xgb_model = xgb.XGBRegressor(tree_method="gpu_hist", random_state=10)
xgb_model = xgb.XGBRegressor(tree_method="auto", random_state=10)

bayes_search = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_iter=20,
        n_jobs=-1,
        verbose=3,
        random_state=0)

print("Training Starts")
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
    verbose=False,
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
    n_jobs=1,
    verbose=3,
    random_state=0
)
print("Training Starts with CatBoost on 4 GPUs")
# %%
print("Training Starts")
with timer():
    bayes_search.fit(train_X, train_y)
    print("Training Finishes")
# %%
# bayes_search.fit(train_X, train_y)
# print("Training Finishes")
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
#print("FAIL(Post Processing)!" + " Error: " + str(e))
# %%
best_models
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
# Inferencing
y_pred = best_models.predict(test_25_X)
remaining_months = y_pred/30
print(remaining_months.shape)

actual_remaining = test_25_y/30
final_test_predicted_df = pd.DataFrame({
    "CUST_ACCOUNT_NUMBER": last_rows_X_test.loc[test_25_X.index,"CUST_ACCOUNT_NUMBER"].astype(str),
    "PREDICTED_MONTHS_REMAINING": remaining_months,
    "ACTUAL_MONTH_REMAINING": actual_remaining,
    "RESIDUAL": abs(remaining_months-actual_remaining)
})

final_test_predicted_df.head(50)
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
y_pred = best_models.predict(test_X)
remaining_months = y_pred/30
print(remaining_months.shape)

actual_remaining = test_y/30
final_test_predicted_df = pd.DataFrame({
    "CUST_ACCOUNT_NUMBER": comprehensive_imputed_df.loc[test_X.index,"CUST_ACCOUNT_NUMBER"].astype(str),
    "PREDICTED_MONTHS_REMAINING": remaining_months,
    "ACTUAL_MONTH_REMAINING": actual_remaining,
    "RESIDUAL": abs(remaining_months-actual_remaining)
})
final_test_predicted_df.head(796)
# %%
final_test_predicted_df
# %%
# %%
# Feature Importance Scores
# importance = best_models.get_booster().get_score(importance_type='gain')
feature_importance = best_models.get_feature_importance()
feature_names = best_models.feature_names_

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
# # Data extrcation of Live Customers for Inferencing

# %%
#Code to impute missing usage data for first half of 2023
#Code to impute missing usage data for first half of 2023

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
usage['YYYYWK']
# %%
# %% PREPROCESSING RAW DATA
#  Sproc_5
#  Justin Bovard 2025-07-15
# Fetching all the different views
#Fetching all the different views
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

session.write_pandas(merged_5, "PS_DOCUWARE_RAW_DATA_PREDICTION", auto_create_table=True, overwrite = True)
# %% [markdown]
# # Data Preprocessing of Live customers
#Fetching from previous SPROC output table
raw_df_active = session.sql("SELECT * FROM RUS_AIML.PS_DOCUWARE_RAW_DATA_PREDICTION").to_pandas()

non_ts_numeric_cols = ["PROBABILITY_OF_DELINQUENCY", "RICOH_CUSTOM_RISK_MODEL"]
non_ts_categorical_cols = ["OVERALL_BUSINESS_RISK", "PAYMENT_RISK_TRIPLE_A_RATING", "CONTRACT_LINE_ITEMS"]
columns_to_be_processed_later = non_ts_numeric_cols + non_ts_categorical_cols + ["CUST_ACCOUNT_NUMBER"]
finalized_df_ohe_to_process = raw_df_active.groupby("CUST_ACCOUNT_NUMBER")[columns_to_be_processed_later].first()


# Imputation for Non Time Series columns
pofd_median = finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"].median()
finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"] = finalized_df_ohe_to_process["PROBABILITY_OF_DELINQUENCY"].apply(lambda x:   float(pofd_median) if np.isnan(x) else x)

rcrm_mode = finalized_df_ohe_to_process["RICOH_CUSTOM_RISK_MODEL"].mode()
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
raw_df_active.drop(columns_to_be_droped, axis=1, inplace=True)

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

# Imputation for Time Series columns
ts_columns = ["CUST_ACCOUNT_NUMBER", "YYYYWK", "DOCUMENTS_OPENED", "USED_STORAGE__MB", "INVOICE_REVLINE_TOTAL", "ORIGINAL_AMOUNT_DUE", "FUNCTIONAL_AMOUNT"]
raw_df_active["transformed_YYYYWK"] = raw_df_active["MONTH"].apply(convert_date_to_yyyywk)
#Impute missing YYYYWK with equivalent MONTH
raw_df_active["YYYYWK"].fillna(raw_df_active["transformed_YYYYWK"], inplace=True)
raw_df_active.drop("transformed_YYYYWK", axis=1, inplace=True)
ts_df = raw_df_active[ts_columns]
ts_df = ts_df[ts_df['YYYYWK'].notna()]
ts_df['YYYYWK'] = ts_df['YYYYWK'].astype(int)
ts_df['CUST_ACCOUNT_NUMBER'] = ts_df['CUST_ACCOUNT_NUMBER'].astype(int)
ts_df.rename(columns={'USED_STORAGE__MB':'USED_STORAGE_MB'},inplace=True)
ts_df_sorted = ts_df.sort_values(['CUST_ACCOUNT_NUMBER','YYYYWK']).drop_duplicates()

only_features = ts_df_sorted.copy()
only_features = only_features.fillna(0)

#ts_comprehensive_df = session.sql("SELECT * FROM RAC_RAPID_DEV.RUS_AIML.PS_DOCUWARE_PREDICTION_TS_DF_ONLY").to_pandas()
#training_set_ts_df = session.sql("SELECT * FROM RAC_RAPID_DEV.RUS_AIML.PS_DOCUWARE_ONLY_TS_DF").to_pandas()
#ts_comprehensive_df_active = time_series_ts_fresh_features(only_features)
# %% [markdown]
# # Store Live Customers' TimeSeries data before TSFRESH into DB
session.write_pandas(only_features, "PS_DOCUWARE_LIVE_RAW_DATA_BEFORE_TSFRESH", auto_create_table=True, overwrite = True)
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
"""raw_df = session.sql("SELECT * FROM RAC_RAPID_DEV.RUS_AIML.PS_DOCUWARE_RAW_DATA_PREDICTION").to_pandas()
# CK 5-21 not sure why extraction was used over prediction raw_df = session.sql("SELECT * FROM RAC_RAPID_DEV.RUS_AIML.PS_DOCUWARE_RAW_DATA_EXTRACTION").to_pandas()
df_cust_earliest_date = raw_df[['CUST_ACCOUNT_NUMBER', 'FINAL_EARLIEST_DATE']]
df_cust_earliest_date.drop_duplicates(inplace=True)
df_cust_earliest_date['CUST_ACCOUNT_NUMBER'] = df_cust_earliest_date['CUST_ACCOUNT_NUMBER'].astype(int)
all_usage['CUST_ACCOUNT_NUMBER'] = all_usage['CUST_ACCOUNT_NUMBER'].astype(int)
df_merged = pd.merge(df_cust_earliest_date, all_usage , on='CUST_ACCOUNT_NUMBER', how='inner')"""
# %%
#raw_df.head()

# %%
#df_merged.head()

# %%
#final_features_for_FE

# %%
"""df_merged['CUST_ACCOUNT_NUMBER'] = df_merged['CUST_ACCOUNT_NUMBER'].astype(int)
final_features_for_FE['CUST_ACCOUNT_NUMBER'] = final_features_for_FE['CUST_ACCOUNT_NUMBER'].astype(int)
final_features_for_FE_trimmed = pd.DataFrame()

for ind,row in df_merged.iterrows():
    t = final_features_for_FE[ final_features_for_FE['CUST_ACCOUNT_NUMBER'] == row['CUST_ACCOUNT_NUMBER'] ]
    t['YYYYWK'] = t['YYYYWK'].astype(int)
    yyyywk_date = convert_yyyywk_to_actual_mid_date(row['YYYYWK'])

    row['YYYYWK'] = int(row['YYYYWK'])
    tt = t[ t['YYYYWK'] >= row['YYYYWK'] ]
    tt['DAYS_TO_CHURN'] = tt['DAYS_TO_CHURN'] - (pd.to_datetime(yyyywk_date) - pd.to_datetime(row['FINAL_EARLIEST_DATE'])).days
    final_features_for_FE_trimmed = pd.concat([tt, final_features_for_FE_trimmed], axis=0, ignore_index=True)"""

# %%
#final_features_for_FE_trimmed.head()

# %%
#session.write_pandas(final_features_for_FE_trimmed, "PS_DOCUWARE_LIVE_RAW_DATA_BEFORE_TSFRESH", auto_create_table=True, overwrite = True)

# %% [markdown]
# # Feature Engineering using TSFRESH

# %%
#CK 5-21 ts_comprehensive_df_active = engineer_timeseries_cols_using_tsfresh_for_live_customers(final_features_for_FE_trimmed)
with timer():
    ts_comprehensive_df_active = engineer_timeseries_cols_using_tsfresh_for_live_customers(only_features)
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
session.write_pandas(ts_comprehensive_df_active, "PS_DOCUWARE_LIVE_CUSTOMER_FEATURES", auto_create_table=True, overwrite = True)
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
model_file = os.path.join('/tmp', 'xgb.joblib.gz')
session.file.get("@PS_DOCUWARE_CHURN/ps_docuware_churn_model/xgb.joblib.gz", "/tmp")
best_model = load(model_file)

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
session.write_pandas(df_live_customers, "PS_DOCUWARE_LIVE_CUSTOMER_FEATURES", auto_create_table=True, overwrite = True)
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
final_predicted_df

# %% [markdown]
# # Store Prediction for Live customers into DB
# %%
session.write_pandas(final_predicted_df, "PS_DOCUWARE_PREDICTION_DATA_NEW", auto_create_table=True, overwrite = True)
# changed != to =
session.sql("INSERT INTO rus_aiml.PS_DOCUWARE_PREDICTION_DATA_ARCHIVE SELECT * FROM rus_aiml.PS_DOCUWARE_PREDICTION_DATA_NEW WHERE CREATION_DATE not in (select creation_date from rus_aiml.PS_DOCUWARE_PREDICTION_DATA_ARCHIVE)")
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


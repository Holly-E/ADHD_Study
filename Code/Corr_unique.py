# -*- coding: utf-8 -*-
"""
Erickson, Holly
ADHD STUDY
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/slaits/nsch_2011_2012/04_List_of_variables_and_frequency_counts/create_formatted_frequencies.pdf
"""

import numpy as np
import pandas as pd
#import seaborn as sn
#import matplotlib.pyplot as plt

np.random.seed(123)

#%%
"""
Read file with highest corr for ADHD with and without K2Q34A
"""

file_path1 = 'C://Master/Semester_6/Github/ADHD_Study/Code/Target_corr_comparison.csv'
df1 = pd.read_csv(file_path1)
cols1 = df1.columns

#%%
print(cols1)
conduct = df1['feature - only conduct'].values
no_conduct = df1['feature - no conduct'].values

#%%
only_conduct = [x for x in conduct if x not in no_conduct]
both = [x for x in conduct if x in no_conduct]
no_conduct_cleaned = [x for x in no_conduct if str(x) != 'nan']
only_no_conduct = [x for x in no_conduct_cleaned if x not in conduct]
#%%
df = pd.DataFrame()
df['High corr in both'] = both

for x in range(len(both)):
    if x in range(len(only_conduct)):
        pass
    else:
        only_conduct.append(np.nan)
    if x in range(len(only_no_conduct)):
        pass
    else:
        only_no_conduct.append(np.nan)
#%%

df["only_conduct"] = only_conduct
df["only_no_conduct"] = only_no_conduct

df.to_csv("Corr features.csv", index = False)

#%%
"""
create a dictionary of df1 :
    - only conduct feature : corr
    - only no conduct feature: corr
"""
corr_dict = {
    "both_corrs": {},
    "cor_cond": [],
    "no_cond": []
    }  

for row in range(len(df1)):
    feature_conduct = df1["feature - only conduct"][row]
    feature_no_conduct = df1["feature - no conduct"][row]
    
    if feature_conduct in df["High corr in both"].values: # corr_dict[]
        corr_dict.both_corrs[feature_conduct] = [df.index[df['BoolCol']].tolist()] #"Corr subset"]
    #corr_dict[feature_conduct] = df1["Corr_subset"][row]
   # corr_dict[feature_no_conduct] = df1["Corr_subset - no conduct"]
    
#%%
"""
row_1 = [] # row index for df1 feature - only conduct
row_2 = [] # row index for df1 feature - no conduct
corr_1 = [] # correlation val for feature - only conduct
corr_2 = [] # correlation val for feature - no conduct

for row in range(len(df)):
    feat1 = df["High corr in both"][row]
    row_1.append() # index of df1 only conduct feat
    row_2.append() # index of df1 only no conduct feat
    
    # corr_1 
    feat2 = df["only_conduct"][row]
    feat3 = df["only_no_conduct"][row]
    
#%%
    
    corr_1.append()
    corr_2.append
    corr_3.append
    corr_4.append
"""


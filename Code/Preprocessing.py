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
read data
"""
file_path = 'C://Master/Semester_6/Github/ADHD_Study/Code/nsch_2011_2012_puf/nsch_2011_2012_puf.sas7bdat'
df = pd.read_sas(file_path)

#%%
cols = df.columns
cols1 = [x[0:3] for x in cols]
df1 = pd.DataFrame()
df1['cols'] = cols1
print(df1.cols.value_counts())
#%%
"""
9 functions
"""
# REMOVE ROWS: 
# Keep rows based on value in column 
def keep_rows(search_col, good_vals, df):
    """
    keeps relevant rows based on value in seach column
    search_col = val
    returns df
    """
    new_df = df.loc[df[search_col].isin(good_vals)]
    print(df[search_col].value_counts())
    new_df.info()
    return new_df

# Make sure target col has acceptable vals
def target_01(df):
    search_col = 'K2Q31A'
    good_vals = [0,1]
    new_df = keep_rows(search_col, good_vals, df)
    return new_df

# REMOVE COLUMNS (Split Y first if needed):
# Keep relevant columns in df
def keep_cols(keep_col, df):
    """
    keeps relevant columns based on value in seach column
    keep_col = list
    returns df
    """
    new_df = df.drop([x for x in df.columns if x not in keep_col], axis = 1)
    print("Length of original df:")
    print(len(list(df)))
    print("Length new df:")
    print(len(list(new_df)))
    return new_df

# Keep features based on codes in a df
def cols_from_df(df_codes, codes, df):
    """
    Parameters
    ----------
    df_codes : dataframe
    df_codes: df
        Variable: features column
        Codes: keep codes
    codes : list of codes to keep

    Returns df with only columns that are in codes (could also do a join & drop)

    """
    codes_column = "Codes" 

    codes_keep = df_codes[df_codes[codes_column].isin(codes)] # df - keeps rows with code
    dropped = []
    for col in df.columns:
        if col not in codes_keep.Variable.values:
            if col != 'SEX':
                dropped.append(col)
    new_df = df.drop(dropped, axis = 1, inplace = False)
    return new_df

# Hide 6,7 vals by making them NaN 
def hide_67(df):
    """
    Update responses for unknown and refused according to dict_resp: 
    Unknown answers were coded as “6,” “96,” or “996”  
    Refused responses were coded as “7,” “97,” or “997”  
    """
    cols = df.columns

    df_replaced = pd.DataFrame()
    for col in cols:
        replace = []
        vals = df[col].value_counts()
        len_answers = len(vals)
        if len_answers <= 7:
            replace = [6, 7]
            df_replaced[col] = df[col].replace(replace,[np.NaN,np.NaN])
        elif len_answers <= 100:  # leave["IDNUMR", "K2Q04R", "NSCHWT"] "
            replace = [96, 97]
            df_replaced[col] = df[col].replace(replace,[np.NaN,np.NaN])
        else:
            df_replaced[col] = df[col]
        
    return df_replaced

# Normalize data
def norm_df(df):
    """
    Returns Normalized df
    """
    from sklearn import preprocessing
    vals = preprocessing.normalize(df)
    new_df = pd.DataFrame(data = vals, columns = list(df))
    return new_df

# Impute values for NaN
def impute_vals(df, strategy_used = 'most_frequent'):
    """
    Simple imputer
    """

    # Remove columns with all NaNs
    # Uncomment out if you want the list of which columns were dropped; function works the same.
    """
    headers = x.columns.values
    empty_train_columns =  []
    for col in headers:
    # all the values for this feature are null
    if sum(x[col].isnull()) == x.shape[0]:
    empty_train_columns.append(col)
    print(empty_train_columns)
    """
    have_vals = df.dropna(axis=1, how='all', inplace=False)
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy= strategy_used)
    imputer.fit(have_vals)
    vals = imputer.transform(have_vals)
    
    new_df = pd.DataFrame(data = vals, columns = list(have_vals))
    return new_df

# Save DF to csv
def save_df(name, df):
    # Used by y target split function
    df.to_csv("{}.csv".format(name), index = False)

# Split off Y Target, returns x and saves Y
def y_split(KEEP, df):
    """
    returns x df
    """
    Y = keep_cols(['K2Q31A'], df) 
    # Save Y
    name = "Y{}".format(KEEP)
    save_df(name, Y)
    x = keep_cols([x for x in list(df) if x not in ['K2Q31A']], df)
    return x

#%%
"""
all rows 
"""
df = pd.read_sas(file_path)
KEEP = 23
name = "x_keep_{}".format(KEEP)


df = target_01(df)
df = hide_67(df)
df = impute_vals(df)

x = y_split(KEEP, df)

# Remove columns and adjust vals
df_codes = pd.read_csv("Corr features codes.csv")
codes = [3]
x = cols_from_df(df_codes, codes, x)

save_df(name, x)
print("Final DF:")
print(len(list(x)))
#%%

"""
only female rows 
"""
df = pd.read_sas(file_path)
KEEP = 20
name =  "x_keep_{}_w_conduct_female".format(KEEP)

search_col = 'SEX'
good_vals = [2]
df = keep_rows(search_col, good_vals, df)

df = target_01(df)
df = hide_67(df)
df = impute_vals(df)

x = y_split(KEEP, df)

# Remove columns and adjust vals
df_codes = pd.read_csv("Corr features codes.csv")
codes = [3]
x = cols_from_df(df_codes, codes, x)

save_df(name, x)
print("Final DF:")
print(len(list(x)))
#%%
"""
only feats from Corr Features Codes 3, only K2Q34A = 0 rows
"""
df = pd.read_sas(file_path)
KEEP = 22
name = "x_keep_{}_only_conduct_corr_feats_3".format(KEEP)

# Remove rows and split y
search_col = 'K2Q34A'
good_vals = [1]
df = keep_rows(search_col, good_vals, df)

df = target_01(df)
x = y_split(KEEP, df)

# Remove columns and adjust vals
df_codes = pd.read_csv("Corr features codes.csv")
codes = [3]
x = cols_from_df(df_codes, codes, x)

x = hide_67(x)
x = impute_vals(x)

save_df(name, x)
print("Final DF:")
print(len(list(x)))

#%%
"""
Unsupervised model based on 3 attributes
"""
df = pd.read_sas(file_path)
KEEP = 21
name = "keep_{}_w_conduct_only_ADHD_w_cluster_labels".format(KEEP)
#%%
# Rows Only ADHD = 1
search_col = 'K2Q31A'
good_vals = [1]
df = keep_rows(search_col, good_vals, df)

# Columns only attributes
keep_col = ["K2Q10","K2Q16", "K2Q32A", "K7Q84", "K7Q83", "K7Q82", "K8Q21", "K8Q34", "K2Q34A"]
df = keep_cols(keep_col, df)

#remove rows for nan in 3 attributes
df.dropna(axis = 0, inplace = True)

df = hide_67(df)
#df = norm_df(df)


from itertools import combinations 
  
comb_list = []
# Get all combinations and all lengths
# Forward step-wise regression to choose features for k-means:

# 1. Perform k-means on each of the features individually for some k.
# 2. For each cluster measure Davies boulding score (lower is better)

from sklearn.cluster import KMeans 
from sklearn.metrics import davies_bouldin_score 

master = {}
for length in range(6):
    if length > 2:
        score = 10
        lf = []  #List of features selected
        print("Length is {}".format(length))
        comb = combinations(keep_col, length) 
        # Print the obtained combinations 
        for i in list(comb): 
            print (i)
            comb_list.append(i)

            X = keep_cols([x for x in i], df)
            X.dropna(axis = 0, inplace = True)
            kmeans = KMeans(n_clusters=3, random_state=1).fit(X) 
            
            # cluster labels 
            labels = kmeans.labels_ 
            dbs = davies_bouldin_score(X, labels)
            
            print(dbs) 
            
            # 3. Take the features which gives the best performance (lowest DBS) and add it to Sf
            if dbs == score:
                print ("same")
            elif dbs < score:
                score = dbs
                lf = i
        master[length] = [score, lf]

#%%     
for val in [3, 4, 5]:
    """
    use elbow method to determine optimal k for each set of features
    """
    current_feats = master[val][1]
    print(current_feats)
    current_df = keep_cols([x for x in current_feats], df)
    current_df.dropna(axis = 0, inplace = True)
    
    import matplotlib.pyplot as plt
    Sum_of_squared_distances = []
    K = range(1,7)
    for k in K:
        km = KMeans(n_clusters=k, n_jobs = -1)
        km = km.fit(current_df)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title(val)
    plt.show()

#%%
current_feats = master[5][1]
x = keep_cols([x for x in current_feats], df)
x.dropna(axis = 0, inplace = True)
kmeans = KMeans(n_clusters=3, random_state=0, n_jobs=-1).fit(x)
labels = kmeans.labels_
#label_df = pd.DataFrame(labels, columns = ['labels'])

x_cols = list(x)
x['label'] = labels
print(x['label'].value_counts())

centroids = kmeans.cluster_centers_
centroid_df = pd.DataFrame(data = centroids, columns = x_cols)

centroid_df.to_csv('keep_{}_centroids_5_feat_3_clus.csv'.format(KEEP), index = False)

save_df(name, x)

#%%
"""
TEMPLATE
CTRL 1 to uncomment
"""
# df = pd.read_sas(file_path)
# KEEP = 'keepnumber'
# name = "keep_{}_REPLACE".format(KEEP) # OR x_keep if splitting


# # Adjust rows and split y
# # x = y_split(KEEP, df)


# # Adjust cols
# keep_col = []
# df = keep_cols(keep_col, df)

# print("Final DF:")
# save_df(name, df)
# print(len(list(df)))

# save_df(name, x)
# print(len(list(x)))

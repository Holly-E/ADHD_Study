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
Read File without replacing unkown and refused with -1, -2
Read SAS file using pandas
"""

file_path1 = 'C://Master/Semester_6/Github/ADHD_Study/Code/nsch_2011_2012_puf/nsch_2011_2012_puf.sas7bdat'
df1 = pd.read_sas(file_path1)
cols1 = df1.columns

"""
Update responses for unknown and refused according to dict_resp: 
Unknown answers were coded as “6,” “96,” or “996”  
Refused responses were coded as “7,” “97,” or “997”  
"""
dict_resp = {
    "unknown": -1,
    "refused": -2
    }
df1_replaced = pd.DataFrame()
for col in cols1:
    replace = []
    vals = df1[col].value_counts()
    len_answers = len(vals)
    if len_answers <= 7:
        replace = [6, 7]
        df1_replaced[col] = df1[col].replace(replace,[-1,-2])
    elif len_answers <= 100:  # leave["IDNUMR", "K2Q04R", "NSCHWT"] "
        replace = [96, 97]
        df1_replaced[col] = df1[col].replace(replace,[-1,-2])
    else:
        df1_replaced[col] = df1[col]

df1_replaced.to_csv('df1_replaced.csv', index = False)

#%%

"""
Read File without replacing unkown and refused with -1, -2
Started using df1_replaced at 13th round
"""

df1 = pd.read_csv('OUTPUT_x_keep_6_no_conduct_code_2RF_Feat_imp.csv')
cols1 = df1.columns

#%%
"""
Target K2Q31A = Has a doctor or other health care provider ever told you that [S.C.] had
Attention Deficit Disorder or Attention-Deficit/Hyperactivity Disorder, that is, ADD or ADHD?
                        Frequency   Percent     Frequency_VI   Percent_VI
L - LEGITIMATE SKIP     10040       10.49       174            7.43 
M - MISSING IN ERROR    10          0.01        1              0.04 
0 - NO                  76982       80.46       2087           89.11
1 - YES                 8528        8.91        77             3.29
6 - DON'T KNOW          89          0.09        3              0.13 
7 - REFUSED             28          0.03        0              0.0

"""
target = 'K2Q31A'
print(df1['K2Q31A'].value_counts())
#print(df2['K2Q31A'].value_counts())

"""
OUTPUT if using df1_replaced:
 0.0    76982
 1.0     8528
-1.0       89
-2.0       28
Name: K2Q31A, dtype: int64

OUTPUT:
0.0    76982
1.0     8528
6.0       89
7.0       28
Name: K2Q31A, dtype: int64
0.0    2087
1.0      77
6.0       3
Name: K2Q31A, dtype: int64
"""
#%%
"""
Correlation Matrices
"""
df1_corr = df1.corr()
#df2_corr = df2.corr()

#%%
"""
sn.heatmap(df1_corr, annot=True)
plt.show()


#sn.heatmap(df2_corr, annot=True)
plt.rcParams.update({'font.size': 8})
plt.show()

"""
#%%
"""
Prediction Column:
K2Q31A Attention Deficit Disorder or Attention-Deficit/Hyperactivity Disorder, that is, ADD or ADHD?
HELP SCREEN: A child with Attention Deficit Disorder or Attention Deficit Hyperactive
Disorder has problems paying attention or sitting still. It may cause the child to be easily
distracted.

K2Q31A Responses (DF1)
                    Frequency Percent Frequency Percent Cumulative
L - LEGITIMATE SKIP  10040     10.49     10040      10.49 
M - MISSING IN ERROR    10      0.01     10050      10.50   
0 - NO               76982     80.46     87032      90.96 
1 - YES               8528      8.91     95560      99.88 
6 - DON'T KNOW          89      0.09     95649      99.97 
7 - REFUSED             28      0.03     95677     100.00

Features that we are also interested in:
K2Q31A_1 How old was [S.C.] when you were first told by a doctor or other health care provider that [he/she] had [CONDITION]? 
K2Q31A_2 (Same question as 1 but answer is unit of measure)

K2Q31B Does [S.C.] currently have ADD or ADHD? 
K2Q31C Would you describe [his/her] ADD or ADHD as mild, moderate, or severe? 
K2Q31D Is [S.C.] currently taking medication for ADD or ADHD?


Autism, Depression Anxiety


"""
"""
Add correlation column to a df
"""
df1_corr_target = pd.DataFrame()
df1_corr_target['Corr_subset'] = df1_corr['K2Q31A']
df1_corr_target.drop('K2Q31A', inplace = True)
df1_corr_target['feature'] = df1_corr_target.index

#%%
join_df = df1.merge(df1_corr_target, on='feature', how='left')

join_df.to_csv('OUTPUT_x_keep_6_no_conduct_code_2RF_Feat_imp.csv', index = False)
#%%
"""
keep = []
for col in df1_corr_cols:
    if (df1_corr_target[col][0] >.1) | (df1_corr_target[col][0] < -.1):
        keep.append(col)

df1_corr_strong = df1_corr_target.loc[ : , keep]


df1_corr_strong_T = df1_corr_strong.T
df1_corr_strong_T.sort_values(by = target, inplace = True, ascending = False)
df1_corr_strong_T.to_csv('df1_corr_strong.csv')
"""
#%% 
"""
Read in df1_corr_strong with training codes column added 
"""
codes_column = "Keep in training " 
codes = pd.read_csv('df1_corr_strong_training_codes.csv')


#%%
"""
Codes of "Keep in training " that we wish to keep 

"""

#codes_keep = codes[codes[codes_column].eq(2)]  #44
# Keep multiple codes                   
codes_keep = codes[codes[codes_column].isin([1,2])] #63

#%%            
# If were keeping all except ADHD derived 362
all_cols = list(df1)
adhd_cols = ['K2Q31A_1', 'K2Q31A_2', 'K2Q31B', 'K2Q31C', 'K2Q31D']
codes_keep = [x for x in all_cols if x not in adhd_cols] 

#%%

"""
Remove redundant columns with high correlation.
This will keep only those columns with correlation less than 0.9

"""
#corr = correlation matrix DF from data.corr() 
columns = np.full((df1_corr.shape[0],), True, dtype=bool)
for i in range(df1_corr.shape[0]):
    for j in range(i+1, df1_corr.shape[0]):
        if df1_corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = df1.columns[columns]
clean = df1[selected_columns]

cols = list(clean)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index(target)))
clean = clean.loc[:, cols]

#%%
high_corr = [col for col in cols1 if col not in cols] 
high_corr_df = pd.DataFrame(data = high_corr, columns = ["Removed"])
high_corr_df.to_csv('high_corr.csv', index = False)

#%% 
"""
DO THIS FIRST
1. For training the model, only keep rows with answer of 1 or 0 in target 
2. Impute missing values:
https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/
3. Determine variables to remove because they are based on prior diagnosis of ADHD
4. Run quick and dirty model to get general idea
"""

clean_targeted = clean.loc[clean[target].isin([0,1])]
clean_targeted.info()

"""
<class 'pandas.core.frame.DataFrame'>
Int64Index: 85510 entries, 0 to 95676
Columns: 310 entries, K2Q31A to NSCHWT
dtypes: float64(310)
memory usage: 202.9 MB
"""
#%%
"""
Second Test - keep 3 - remove rows that predict one for 'K2Q34A'  - conduct problems
to see if we can make a model for those that do not have conduct problems 
"""

#clean_targeted_no_conduct = clean_targeted.loc[clean_targeted[ 'K2Q34A'].isin([1])]
clean_targeted_no_conduct = clean_targeted.loc[clean_targeted[ 'K2Q34A'].isin([0,6])]
clean_targeted_no_conduct.info()

# Split into x and Y dfs
x = clean_targeted_no_conduct.iloc[:,1:]
Y = pd.DataFrame()
Y[target] = clean_targeted_no_conduct[target]

#%%
"""
With conduct
"""
# Split into x and Y dfs
x = clean_targeted.iloc[:,1:]
Y = pd.DataFrame()
Y[target] = clean_targeted[target]


#%%
# Remove columns with all NaNs
x_have_vals = x.dropna(axis=1, how='all', inplace=False)

headers = x.columns.values
empty_train_columns =  []
for col in headers:
    # all the values for this feature are null
    if sum(x[col].isnull()) == x.shape[0]:
        empty_train_columns.append(col)
print(empty_train_columns)
# Dropped ['K6Q13A', 'K6Q13B']
# 2nd round, dropped ['K2Q34A_1', 'K2Q34A_2', 'K2Q34B', 'K2Q34C', 'K6Q13A', 'K6Q13B']

#%%
"""
Impute missing values
Used most frequent
"""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(x_have_vals)
xtrans_vals = imputer.transform(x_have_vals)
    
xtrans = pd.DataFrame(data = xtrans_vals, columns = x_have_vals.columns.values )
#%%
"""
Drop IDNUMR col and columns that are not in keep codes 
"""

xtrans.drop('IDNUMR', axis = 1, inplace = True)

        
#%%

dropped = []
for col in xtrans.columns:
    if col not in codes_keep.Variable.values:
    
    # ADHD vars only variation
    #if col not in codes_keep:
        dropped.append(col)
        xtrans.drop(col, axis = 1, inplace = True)


#%%
xtrans.to_csv('x_keep_9_no_conduct_codes_12.csv', index = False)
Y.to_csv('Y9.csv', index = False)

#%%
"""
Group by sex
Find out Number of ADHD diagnosis per sex
"""
df_sex = df1.groupby(['SEX', target]).size()

"""
SEX	K2Q31A	MALE
1.0	0.0	38154
1.0	1.0	5940

SEX	K2Q31A	FEMALE
2.0	0.0	38729
2.0	1.0	2585

12% total males diagnosed with ADHD
5.5% total females diagnosed with ADHD
"""

    
#%%
"""
Research 
- Male - Female ration in / not in behavior concerns (with ADHD)
- Answers to medication questions of those with and without behavior concerns
"""



#%%
"""
Join final features list with codes_keep to get the descriptions
"""
joiner = "x_keep_6_no_conduct_code_2RF_Feat_imp.csv"
df_join = pd.read_csv(joiner)
df_join_desc = df_join.join(codes.set_index(['Variable']), on = ['feature'], how = 'left' )

#%%
df_join_desc.to_csv('OUTPUT_' + joiner, index = False)

#%%
"""
Calculations to DO:
P Value between two columns:
df1_clean = df1.dropna()
stats.pearsonr(df1_clean['VIXCLS'], df1_clean['GDP'])


Selecting the columns based on how they affect the p-value. 
We assume to null hypothesis to be “The selected combination of dependent variables do not have any effect on the independent variable”.
Then we build a small regression model and calculate the p values.
If the p values is higher than the threshold, we discard that combination of features.
First move target to front and remove the nan values K2Q31A because it is the column we are trying to predict
https://stackoverflow.com/questions/44495667/calculate-p-value-in-sklearn-using-python
https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf
"""
"""
selected_columns = selected_columns[1:].values

#%%
#import statsmodels.api  as sm
from sklearn.linear_model import LinearRegression
from regressors import stats
def backwardElimination(x, Y, sl, columns):
    numVars = len(selected_columns)
    print(numVars)
    for i in range(0, numVars):
        model = LinearRegression()
        ols = model.fit(Y, x)
        print("coef_pval:\n", stats.coef_pval(ols, x, Y))
        """
"""
        # Not sure about this if using SKlearn instead of statsmodel
        maxVar = max(regressor.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
       """   
"""
    #regressor.summary()
    #return x, columns

SL = 0.05
data_modeled, selected_columns = backwardElimination(x, Y, SL, selected_columns)

#df1_target = pd.DataFrame()
#df1_target[target] = df1.iloc[:,0]

#df1_train = pd.DataFrame(data = data_modeled, columns = selected_columns)

"""
#%%
"""
Here are the distribution plot for the selected features:
fig = plt.figure(figsize = (20, 25))
j = 0
for i in data.columns:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(data[i][result['diagnosis']==0], color='g', label = 'benign')
    sns.distplot(data[i][result['diagnosis']==1], color='r', label = 'malignant')
    plt.legend(loc='best')
fig.suptitle('Breast Cance Data Analysis')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
"""
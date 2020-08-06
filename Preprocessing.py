# -*- coding: utf-8 -*-
"""
Erickson, Holly
ADHD STUDY
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/slaits/nsch_2011_2012/04_List_of_variables_and_frequency_counts/create_formatted_frequencies.pdf
"""

# Read SAS file using pandas

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
#%%

file_path1 = 'C://Master/Semester_6/Github/ADHD_Study/nsch_2011_2012_puf/nsch_2011_2012_puf.sas7bdat'
df1 = pd.read_sas(file_path1)
cols1 = df1.columns

file_path2 = 'C://Master/Semester_6/Github/ADHD_Study/nsch_2011_2012_vi_puf/nsch_2011_2012_vi_puf.sas7bdat'
df2 = pd.read_sas(file_path2)
cols2 = df2.columns

file_path3 = 'C://Master/Semester_6/Github/ADHD_Study/nsch1112mimp/nsch1112mimp.sas7bdat'
df3 = pd.read_sas(file_path3)
cols3 = df3.columns

file_path4 = 'C://Master/Semester_6/Github/ADHD_Study/nsch1112mimp_vi/nsch1112mimp_vi.sas7bdat'
df4 = pd.read_sas(file_path4)
cols4 = df4.columns

#%%
"""
K2Q31A = Has a doctor or other health care provider ever told you that [S.C.] had
Attention Deficit Disorder or Attention-Deficit/Hyperactivity Disorder, that is, ADD or ADHD?
                        Frequency   Percent     Frequency_VI   Percent_VI
L - LEGITIMATE SKIP     10040       10.49       174            7.43 
M - MISSING IN ERROR    10          0.01        1              0.04 
0 - NO                  76982       80.46       2087           89.11
1 - YES                 8528        8.91        77             3.29
6 - DON'T KNOW          89          0.09        3              0.13 
7 - REFUSED             28          0.03        0              0.0

"""
print(df1['K2Q31A'].value_counts())
print(df2['K2Q31A'].value_counts())

"""
OUTPUT
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
Visualize Correlation Matrices
"""
corrMatrix1 = df1.corr()
sn.heatmap(corrMatrix1, annot=True)
plt.show()

#%%
"""
corrMatrix2 = df2.corr()
sn.heatmap(corrMatrix2, annot=True)
plt.rcParams.update({'font.size': 8})
plt.show()
"""
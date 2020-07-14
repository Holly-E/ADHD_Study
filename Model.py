# -*- coding: utf-8 -*-
"""
Erickson, Holly
ADHD STUDY
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/slaits/nsch_2011_2012/04_List_of_variables_and_frequency_counts/create_formatted_frequencies.pdf
"""

# Read SAS file using pandas

import pandas as pd

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
1. 10.2 Programming Exercise: Fit a Binary Logistic Regression Model to a Dataset

1. Build a Classification Model

Predict the sex of a person based on their age, name, and state

a. Prepare in Input Features
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import OneHotEncoderEstimator
#%%
spark = SparkSession.builder.appName("Week4").getOrCreate()

csv_file_path = 'C://Master/Semester_5/baby-names.csv'
         
df = spark.read.load(
  csv_file_path,
  format="csv",
  sep=",",
  inferSchema=True,
  header=True
)

df.printSchema()

#%%

# StringIndexer along with the OneHotEncoderEstimator to convert the name, state, and sex columns 
#  Use the VectorAssembler to combine the name, state, and age vectors into a single features vector.

categoricalColumns = ['name', 'state', 'sex']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')
stages += [label_stringIdx]
numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in list(set(df.columns)-set(['year', 'count'])) ]

encoder = OneHotEncoderEstimator(
    inputCols=[indexer.getOutputCol() for indexer in indexers],
    outputCols=["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers]
)

assembler = VectorAssembler(
    inputCols=['name_index_encoded', 'state_index_encoded', 'year'],
    outputCol="features"
)

pipeline = Pipeline(stages=indexers + [encoder, assembler])
df_r = pipeline.fit(df).transform(df)

df_r.printSchema()

#%%

#Your final dataset should contain a column called features containing the prepared vector 
#and a column called label containing the sex of the person.

keep = ['features', 'sex_index']
df_final = df_r.select([column for column in df_r.columns if column in keep])
df_final = df_final.withColumnRenamed("sex_index","label")
df_final.printSchema()

#%%
#2. Fit and Evaluate the Model
#Fit the model as a logistic regression model with the following parameters.
# LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8).
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(df_final)

# Provide the area under the ROC curve for the model.
trainingSummary = lrModel.summary
print('Area Under ROC: ' + str(trainingSummary.areaUnderROC))


"""
Output
Area Under ROC: 0.5
"""

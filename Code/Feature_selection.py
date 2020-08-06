# -*- coding: utf-8 -*-
"""
Erickson, Holly
ADHD STUDY
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/slaits/nsch_2011_2012/04_List_of_variables_and_frequency_counts/create_formatted_frequencies.pdf
"""

# Read SAS file using pandas

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

#%%
# Read the data
x = pd.read_csv('x1.csv')
Y = pd.read_csv('Y1.csv')

#%%
np.random.seed(123)
x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size = 0.2)
#%%

svc=SVC() # The default kernel used by SVC is the gaussian kernel
"""
To avoid warning:  A column-vector y was passed when a 1d array was expected.
.values will give the values in an array. (shape: (n,1)
.ravel will convert that array shape to (n, )
"""
svc.fit(x_train, y_train.values.ravel())

#%%
prediction = svc.predict(x_test)
cm = confusion_matrix(y_test, prediction)
sum = 0
for i in range(cm.shape[0]):
    sum += cm[i][i]
    
accuracy = sum/x_test.shape[0]
print(accuracy)

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

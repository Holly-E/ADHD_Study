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
filename = "x_keep_6_no_conduct_code_2"
x = pd.read_csv(filename + ".csv")
Y = pd.read_csv('Y6.csv')

#x_rem_conduct = x.drop('K2Q34A', axis = 1)

#%%
"""
Normalize data - 7th round
"""
from sklearn import preprocessing
normalized_x = preprocessing.normalize(x)

#%%
np.random.seed(123)
x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size = 0.2)
#%%
"""
Random Forest
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=200, 
                               bootstrap = True,
                               max_features = 'sqrt',
                               n_jobs = -1)

#%%
# get predictions for each row using cross validation 
y_pred = cross_val_predict(model, x, Y.values.ravel(), cv=3)
x['predictions'] = y_pred
x['target'] = Y['K2Q31A']

#%%
# Male / female ratio and count in target
# Male / female ratio and count in predictions
# Get ID column from full df with matching # rows

# Seperate out false positives 
# Male / female ratio and count in false positive
# Get list of ID's for false positive females and list for males - save

#%%
"""
SVC
"""
model = SVC() # The default kernel used by SVC is the gaussian kernel

#%%
"""
To avoid warning:  A column-vector y was passed when a 1d array was expected.
.values will give the values in an array. (shape: (n,1)
.ravel will convert that array shape to (n, )
"""
# Fit on training data
model.fit(x_train, y_train.values.ravel())


prediction = model.predict(x_test)
cm = confusion_matrix(y_test, prediction)
sum = 0
for i in range(cm.shape[0]):
    sum += cm[i][i]
    
accuracy = sum/x_test.shape[0]
print(accuracy)

"""
First round - 44 - conduct included - 44 vars - keep code 2 - SVM - Keep 2
0.915 accuracy :)
    
6th round - 44 - First round repeated using Random Forest - Keep 2
0.915

7th round - 44 - First round repeated normalizing feats - SVM - Keep 2
0.915

10th round - 44 - no conduct - keep code 2 - RF - Keep 6
.925
    
Second round - 63 - no conduct - 63 vars - keep code 1,2 - SVM - Keep 3
0.938

Third round - 63 - conduct included - keep code 1,2 - SVM - Keep 4
0.931

Fourth round - 63 - Second round repeated using Random Forest - Keep 3
0.946

Fifth round - 63 - Third round repeated using Random Forest - Keep 4
.939

8th round - 63 - Conduct only - normalizing feats - keep code 1,2 - RF - Keep 5
.766

9th round - 63 - Conduct only - normalizing feats - keep code 1,2 - SVM - Keep 5
.728

11 Round - 301 - conduct included - keep all except ADHD indicators - RF (increased estimators) - Keep 7
0.935

12 Round - 297 - no conduct - keep all except ADHD indicators - RF (200 estimators) - Keep 8
0.944 

13 Round - 63 - no conduct - keep code 1,2 - RF - Keep 9
0.946

14 Round - 63 - no conduct - keep code 1,2 - SVM - Keep 9
0.940

15 Round - 63 - with conduct - only K2Q31A - K-means - Keep 10

16 Round - 74 - no conduct - Kept corr > .9, replace 6, 7 with Nan & impute vals - keep 16 
0.946

17 Round - 1 - with conduct - only K2Q34A, leave 6,7 in feat - SVM - keep 17
0.915
        
18 - 121 - conduct included - keep code 3, 4, 5, only rows where SEX - Male - RF - keep 18
0.926 - RF
0.915 - SVM

19 - 121 - conduct included - keep code 3, 4, 5, only rows where SEX - Female - RF - keep 19
0.958 - RF
0.951 - SVM
 
20 - 70 -  no conduct - keep code 3, 4 - keep 20
0.944 - RF
0.938 - SVM
 
21 - 5 - unsupervised 3 and 2 clusters - keep 21  

Final model - Intrinsic vals in keep codes 3, 4 - One vs Rest instead of train/test 0.2 split
OR use 10th round 
"""

#%%
"""
Get prediction probabilities and ROC AUC - Random Forest
"""
from sklearn.metrics import roc_auc_score

# Probabilities for each class
rf_probs = model.predict_proba(x_test)[:, 1]

# Calculate roc auc
roc_value = roc_auc_score(y_test, rf_probs)
print(roc_value)

"""
Fourth round 
0.9159

Fifth round
0.935

Sixth round
0.8535

8th round
.832

10th round
.808

11th round
0.938

12th round
0.922

16th Round
0.914

18th Round
0.930

19th Round
0.935

20th Round
0.905
"""
#%%
# Extract feature importances
fi = pd.DataFrame({'feature': list(x.columns),
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)

# Display
print(fi.head())

fi.to_csv("keep_20_RF_Feat_imp.csv", index = False)
"""
Fourth Round
        feature  importance
0   AGEYR_CHILD    0.065624
11       K2Q30A    0.058333
10        CSHCN    0.049003
2         K2Q10    0.047798
54        K7Q84    0.045383

Fifth Round
        feature  importance
9         K2Q22    0.057161
10        CSHCN    0.054641
0   AGEYR_CHILD    0.054530
2         K2Q10    0.051306
11       K2Q30A    0.049207

Sixth Round
        feature  importance
0   AGEYR_CHILD    0.105363
5        K2Q34A    0.080269
32        K7Q84    0.064111
29        K7Q70    0.053416
38        K8Q31    0.050289

8th Round
        feature  importance
2         K2Q10    0.079839
38        K4Q23    0.039359
10        CSHCN    0.032511
54        K7Q84    0.032335
0   AGEYR_CHILD    0.029684

10th Round
        feature  importance
0   AGEYR_CHILD    0.126801
32        K7Q84    0.066422
29        K7Q70    0.063228
39        K8Q32    0.055981
38        K8Q31    0.053462

11th Round
  feature  importance
29   K2Q22    0.045498
17   K2Q10    0.037985
31   CSHCN    0.037476
32  K2Q30A    0.036477
37  K2Q34A    0.028107

12th Round
    feature  importance
32   K2Q30A    0.042085
31    CSHCN    0.041344
17    K2Q10    0.035132
29    K2Q22    0.032271
118   K4Q22    0.021153

18th Round (Males)
   feature  importance
15   K2Q22    0.057678
4    K2Q10    0.053141
17   CSHCN    0.045735
61   K4Q22    0.045082
18  K2Q30A    0.038948

19th Round (Females)
        feature  importance
18       K2Q30A    0.059301
15        K2Q22    0.039905
17        CSHCN    0.038626
4         K2Q10    0.035097
0   AGEYR_CHILD    0.034318

20th Round - 70 (3, 4)
        feature  importance
0   AGEYR_CHILD    0.085158
16       K2Q30A    0.056327
15        CSHCN    0.054334
57        K7Q84    0.049636
2         K2Q10    0.048442
"""

#%%

"""
Get precision - recall score - SVM
"""
from sklearn.metrics import average_precision_score
y_score = model.decision_function(x_test)
average_precision = average_precision_score(y_test, y_score)
print(average_precision)
print(cm)

#%%
#print(y_test.K2Q31A.value_counts())
#%%
"""
DF with x_text, y_test, y_score 
"""
#prediction_df = pd.DataFrame(prediction, columns = ['prediction'])
#y_score_df = pd.DataFrame(y_score, columns = ['y_score'])
#df1_with_scores = pd.concat([y_test, prediction_df, y_score_df,
                             
#concat_list = ['y_test', 'prediction_df', 'y_score_df']

#x_test['y_test'] = y_test
#x_test['prediction'] = prediction
#x_test['y_score'] = y_score
x_test.to_csv('RF_x_keep_16_no_conduct_code_12.csv', index = False)

#%%

list_in_order = ['prediction','y_test', 'y_score']
for ind in range(len(list(x_test)) - 3):
    list_in_order.append(list(x_test)[ind])
  
output = pd.DataFrame(x_test, columns = list_in_order)

#%%
#df1_scores = x_test.drop(['prediction_df', 'y_score_df'], axis = 1)

#Modify keep codes if used (1, 2, 3 currently have been labeled)
output.to_csv(filename + ".csv", index = False)
"""
Round 1 - keep 2
0.4721931323891286
            Predict 0  Predict 1
Actual 0    [[15171     198]
Actual 1     [ 1249   484]]

0.0    15369
1.0     1733

Predicted ADHD and s.c. has not been diagnosed ADHD:
   198 out of 17,102 = 0.0115 (1.15% flag - screen for ADHD out of total pop)
   198 out of 15,369 = 0.012 (1.2% flag - screen for ADHD out of non-diagnosed subset)
   
Did not predict cases that are diagnosed ADHD:
    1249 out of 17,102 total rows = 0.073 (7.3% did not predict prior diagnosis in total pop
    1249 out of 1,733 total ADHD = 0.72 (72% did not predict prior diagnosis in ADHD subset)
    
Predicted not ADHD and s.c. has not been diagnosed ADHD:
    15171 out of 15369 total not ADHD = .987 (98.7% of time - correct that no prior diagnosis)
    
Predicted ADHD and s.c. is indeed diagnosed AHDH:
    484 out of 682 total predicted 1 = .71 (71% of time - correct that prior diagnosis)
    484 out of 17,102 total rows = .028 (2.8% - correctly identified prior diagnosis out of total pop
    484 out of 1,733 total ADHD diagnosis = .279 (28% - correctly identified prior diagnosis out of ADHD subset)

Round 2 - Keep 3
0.5787024396125225
[[15125    86]
 [  934   313]]

Round 3 - Keep 4
0.6977065992880424
[[15187   182]
 [  997   736]]

Round 9 - Keep 5
0.8810918031922734
[[ 35 169]
 [  6 434]]

Round 17 - keep 17
0.26987098799124754
[[15171   198]
 [ 1251   482]]

Round 18 - males 121
0.741
[[7486  135]
 [ 618  580]]

Round 18 - females 121
0.633
[[7722   29]
 [ 372  140]]

Round 20 - 70 feats
0.550
[[15170    67]
 [  949   262]]
"""
#%%
"""
Examine characteristics of 1.15% that are flagged to screen for ADHD
(Actual Target = 0, Predicted = 1)


Distribution plot for the selected features:
    - Plot for Actual 0, view distinction by prediction
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





#%%
"""
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

"""
Output
Area Under ROC: 0.5
"""
#%%
not_norm = pd.read_csv('keep_15_no_conduct_only_ADHD_code_2.csv')
x = pd.read_csv('keep_15_NORM_no_conduct_only_ADHD_code_2.csv')

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k, n_jobs = -1)
    km = km.fit(x)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#%%
kmeans = KMeans(n_clusters=3, random_state=0, n_jobs=-1).fit(x)
labels = kmeans.labels_
#label_df = pd.DataFrame(labels, columns = ['labels'])

x_cols = list(x)
not_norm['label'] = labels
print(not_norm['label'].value_counts())

centroids = kmeans.cluster_centers_
centroid_df = pd.DataFrame(data = centroids, columns = x_cols)

centroid_df.to_csv('centroids_3_keep_15.csv', index = False)

#%%
"""
Plot K2Q34A and K6Q06 colored by label
"""
import seaborn
import matplotlib.pyplot as plt
seaborn.set(style='ticks')

fg = seaborn.FacetGrid(data=not_norm, hue='label', col = 'SEX', row= 'K2Q34A', aspect=1.61)
fg.map(plt.plot, 'K6Q06', 'K2Q34A').add_legend()

#g = sns.FacetGrid(tips, col="time",  row="smoker")
#g = g.map(plt.scatter, "total_bill", "tip", edgecolor="w")

#%%
not_norm.to_csv('keep_11_2_clus_w_conduct_only_ADHD_codes_12.csv', index = False)
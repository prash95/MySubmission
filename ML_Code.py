import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier, MLPRegressor
import matplotlib.pyplot as plt

#Import Training Data and Normalise the Data
training_data = pd.read_excel("dataset/training1.xlsx", header=None)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(training_data)
X1 = scaler.transform(training_data)
X1 = pd.DataFrame(X1)


training_data = pd.read_excel("dataset/training2.xlsx", header=None)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(training_data)
X2 = scaler.transform(training_data)
X2 = pd.DataFrame(X2)

training_data = pd.read_excel("dataset/training3.xlsx", header=None)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(training_data)
X3 = scaler.transform(training_data)
X3 = pd.DataFrame(X3)

#Import Label data for training and normalise the data

training_data = pd.read_excel("dataset/training1_bsi.xlsx", header=None)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(training_data)
y1 = scaler.transform(training_data)
y1 = pd.DataFrame(y1)


training_data = pd.read_excel("dataset/training2_bsi.xlsx", header=None)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(training_data)
y2 = scaler.transform(training_data)
y2 = pd.DataFrame(y2)

training_data = pd.read_excel("dataset/training3_bsi.xlsx", header=None)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(training_data)
y3 = scaler.transform(training_data)
y3 = pd.DataFrame(y3)


#Assign Headers for Data Frames 
for x in range(1,4):
    exec("X"+str(x)+".columns = ['X1','X2','X3','X4','X5']")
for y in range(1,4):
    exec("y"+str(y)+".columns = ['y1']")

#Feature Selection for Training Data
#Backward Elimination using Ordinary Least Squares Model
cols = list(X1.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X1[cols]
 #Adding constant column of ones, mandatory for sm.OLS model
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y1,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print('The selected features for Training Dataset 1 is',selected_features_BE)

#Repeat for all the remaining 2 datasets
cols = list(X2.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_2 = X2[cols]
 #Adding constant column of ones, mandatory for sm.OLS model
    X_2 = sm.add_constant(X_2)
    model = sm.OLS(y2,X_2).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print('The selected features for Training Dataset 2 is',selected_features_BE)

cols = list(X3.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_3 = X3[cols]
 #Adding constant column of ones, mandatory for sm.OLS model
    X_3 = sm.add_constant(X_3)
    model = sm.OLS(y3,X_3).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print('The selected features for Training Dataset 3 is',selected_features_BE)

#Dropping the constant column which was mandatory for Linear Model
X_1 = X_1.drop(['const'], axis = 1)
X_2 = X_2.drop(['const'], axis = 1)
X_3 = X_3.drop(['const'], axis = 1)


#Convert training data to NumPY array()
X1 = X_1.to_numpy()
X2 = X_2.to_numpy()
X3 = X_3.to_numpy()
#Convert Label Data to NumPY array()
y1 = y1.to_numpy()
y2 = y2.to_numpy()
y3 = y3.to_numpy()

x_data = [X1,X2,X3]
y_data = [y1,y2,y3]

#Print Pearson Co-efficient to evaluate the Trained ML model
print ("\tLearning Sheet Number \t" + "Pearson Coefficient")
#repeat stuff for all data
i = 0
j =1
#Train the Artificial Neural Network and Visualise Actual(Label) Vs Predicted
while i < 3:
    fig = plt.figure(j, figsize=(8, 6))
    clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
                         hidden_layer_sizes=(5, 2), random_state=1,max_iter = 3000)
    clf.fit(x_data[i], y_data[i].ravel())
    prediction = clf.predict(x_data[i])
    time = range(0,y_data[i].size)

    print ("\t"+str(i+1) + " \t \t \t " + str(np.corrcoef(prediction, y_data[i].ravel())[0, 1]))
    plt.scatter(time, prediction, color='red', label='''New Index form Datasheet '''+str(i+1))
    plt.scatter(time, y_data[i], color='blue', label="BIS")
    plt.legend()

    plt.xlabel('Time in Seconds')

    plt.ylabel('Training Datasheet ' + str(i+1))
    plt.title('EEG Data Analysis of Datasheet ' + str(i+1) + " Using ANN Algorithm - Training Graph")


    plt.xticks(())
    plt.yticks(())
    j=j+1
    i = i+1
plt.show()
# end repeat

#Import Test Data
X1 = pd.read_excel("dataset/test1.xlsx", header=None)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X1)
X1 = scaler.transform(X1)
X1 = pd.DataFrame(X1)


X2 = pd.read_excel("dataset/test2.xlsx", header=None)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X2)
X2 = scaler.transform(X2)
X2 = pd.DataFrame(X2)


X3 = pd.read_excel("dataset/test3.xlsx", header=None)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X3)
X3 = scaler.transform(X3)
X3 = pd.DataFrame(X3)

#Import Testing Label data(BSI)

testing_data = pd.read_excel("dataset/testing1_bsi.xlsx", header=None)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(testing_data)
y1 = scaler.transform(testing_data)
y1 = pd.DataFrame(y1)


testing_data = pd.read_excel("dataset/testing2_bsi.xlsx", header=None)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(testing_data)
y2 = scaler.transform(testing_data)
y2 = pd.DataFrame(y2)



testing_data = pd.read_excel("dataset/testing3_bsi.xlsx", header=None)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(testing_data)
y3 = scaler.transform(testing_data)
y3 = pd.DataFrame(y3)

#Apply Headers for the DataFrames
for x in range(1,4):
    exec("X"+str(x)+".columns = ['X1','X2','X3','X4','X5']")
for y in range(1,4):
    exec("y"+str(y)+".columns = ['y1']")
    
#Feature Selection for Testing Data
#Backward Elimination using Ordinary Least Squares Model
#Backward Elimination
cols = list(X1.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X1[cols]
 #Adding constant column of ones, mandatory for sm.OLS model
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y1,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print('The selected features for Testing Dataset 1 is',selected_features_BE)

#Repeat for all the remaining 2 datasets
cols = list(X2.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_2 = X2[cols]
 #Adding constant column of ones, mandatory for sm.OLS model
    X_2 = sm.add_constant(X_2)
    model = sm.OLS(y2,X_2).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print('The selected features for Testing Dataset 2 is',selected_features_BE)

cols = list(X3.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_3 = X3[cols]
 #Adding constant column of ones, mandatory for sm.OLS model
    X_3 = sm.add_constant(X_3)
    model = sm.OLS(y3,X_3).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print('The selected features for Testing Dataset 3 is',selected_features_BE)

#Convert Test Data to NumPY()
test_data1 = X_1.to_numpy()
test_data2 = X_2.to_numpy()
test_data3 = X_3.to_numpy()

#Convert Test Label Data to NumPY()
#Import Label
y1 = y1.to_numpy()
y2 = y2.to_numpy()
y3 = y3.to_numpy()


test_data = [test_data1, test_data2, test_data3]
test_data_label = [y1,y2,y3]
#Calculate Pearson Co-efficient for Predicted Testing data against the Actual Values to evaluate performance
print ("\tTesting Sheet Number \t" + "Pearson Coefficient")
i = 0
j=1
while i < 3:
    fig = plt.figure(j, figsize=(8, 6))
    #Using the Trained model stored in clf variable 
    clf.fit(test_data[i],test_data_label[i].ravel())
    prediction = clf.predict(test_data[i])
    time = range(0,prediction.size)
    time1 = range(0,test_data_label[i].size)

    print ("\t"+str(i+1) + " \t \t \t " + str(np.corrcoef(prediction, test_data_label[i].ravel())[0, 1]))
    plt.scatter(time, prediction, color='red', label="Prediction About Test Data")
    plt.scatter(time1, test_data_label[i], color='blue', label="BIS(Label)")
    plt.legend()
    plt.xticks(())
    plt.yticks(())

    plt.xlabel('Time in Seconds')

    plt.ylabel('Testing Datasheet ' + str(i+1))
    plt.title('EEG Data Analysis of Datasheet ' + str(i+1) + " Using ANN Algorithm - Testing Graph")
    i = i + 1
    j = j+1
plt.show()



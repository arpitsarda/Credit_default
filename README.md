1) Imported the liibraries
2) Dropped the data columns of mixed types as they would cause errors while fitting the model
3) Seperated the features in a training file and the target variable in a target file
4) filled the missing values using imputer
5) scaled the data to make sure it comes within 0 to 1
6) applied isolation forest to remove any outliers present in the data as they may distort the findings
7) Trained random forest model, with 1200 estimators
8) Imported the test data file and predcited the values of probablity using it

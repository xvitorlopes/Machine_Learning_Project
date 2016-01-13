# Weight Lifting Exercise

## Summary

This project is to build a machine learning algorithm to recognise different activity quality by using the measurements recorded by the sensors.  
We use the data set here:  
training data (https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)  
test data (https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)    
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.  
We devided pml-training.csv into two parts, one for training and the other for cross validation. We removed variables that are not sensor measures and that consist mostly of NAs and blank. So 52 variables are used as predictors. The "classe" variable is the outcome variable. Both trees and random forests model are built. We adopt the random forests model which has higher accuracy on the cross validation data set. At last we used the random forests model to predict 20 different test cases in the testing set.




## Data Process and Results
1.We read the original training data file, clean the data and split into a trainging set and a cross validation set.

```r
# read the orignal training data file
data <- read.csv("pml-training.csv", na.strings = c("NA", ""))
# remove columns that consist mostly of NAs and blanks
newdata <- data[, colSums(!is.na(data)) == nrow(data)]
# remove columns
# X,user_name,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp,new_window,num_window,
# which are not sensor measurement therefore leaves 53 columns (52
# predictors and 1 outcome)
newdata <- subset(newdata, select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, 
    cvtd_timestamp, new_window, num_window))
# split the data set into training set and cross validation set
library(caret)
inTrain <- createDataPartition(y = newdata$classe, p = 0.7, list = FALSE)
training <- newdata[inTrain, ]  #13737 obs.
validation <- newdata[-inTrain, ]  #5885 obs.   
```

2.We build the model and use the validation data set to calculate the out of sample error.

```r
# predicting with trees
Tree_Fit <- train(classe ~ ., data = training, method = "rpart")
```

```
## Loading required package: rpart
```

```r
# calculate out of sample error
confusionMatrix(validation$classe, predict(Tree_Fit, validation))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1524   20  124    0    6
##          B  479  369  291    0    0
##          C  482   29  515    0    0
##          D  432  181  351    0    0
##          E  151  149  302    0  480
## 
## Overall Statistics
##                                         
##                Accuracy : 0.491         
##                  95% CI : (0.478, 0.504)
##     No Information Rate : 0.521         
##     P-Value [Acc > NIR] : 1             
##                                         
##                   Kappa : 0.334         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.497   0.4933   0.3253       NA   0.9877
## Specificity             0.947   0.8501   0.8812    0.836   0.8885
## Pos Pred Value          0.910   0.3240   0.5019       NA   0.4436
## Neg Pred Value          0.633   0.9201   0.7802       NA   0.9988
## Prevalence              0.521   0.1271   0.2690    0.000   0.0826
## Detection Rate          0.259   0.0627   0.0875    0.000   0.0816
## Detection Prevalence    0.284   0.1935   0.1743    0.164   0.1839
## Balanced Accuracy       0.722   0.6717   0.6033       NA   0.9381
```

Trees model has 49.1% out of sample accuracy, or 50.9% out of sample error.


```r
# predicing with random forests
RF_Fit <- train(classe ~ ., data = training, method = "rf")
# calculate out of sample error
confusionMatrix(validation$classe, predict(RF_Fit, validation))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B   10 1122    6    1    0
##          C    0    3 1020    2    1
##          D    0    0    4  960    0
##          E    0    0    2    4 1076
## 
## Overall Statistics
##                                         
##                Accuracy : 0.994         
##                  95% CI : (0.992, 0.996)
##     No Information Rate : 0.286         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.993         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.994    0.997    0.988    0.993    0.999
## Specificity             1.000    0.996    0.999    0.999    0.999
## Pos Pred Value          1.000    0.985    0.994    0.996    0.994
## Neg Pred Value          0.998    0.999    0.998    0.999    1.000
## Prevalence              0.286    0.191    0.175    0.164    0.183
## Detection Rate          0.284    0.191    0.173    0.163    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.997    0.997    0.994    0.996    0.999
```

Random forests model has 99.4% out of sample accuracy, or 0.6% out of sample error.

3.Random forests appear to have higher out of sample accuracy than trees. We apply random forest to the testing set.

```r
# read the testing set
testing <- read.csv("pml-testing.csv", na.strings = c("NA", ""))
# preprocess the testing set
testProc <- subset(testing, select = names(newdata[, -53]))
# predict the class
answers <- predict(RF_Fit, testProc)
```

4.Generate the submission file.

```r
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}
pml_write_files(answers)
```


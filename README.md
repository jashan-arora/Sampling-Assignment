# Sampling Assignment
### Jashan Arora 102003206 3COE9
## Dataset
The dataset we have consists of **773 rows and 31 columns** and is specifically designed for predicting credit card fraud. It contains 30 input features and 1 output feature, where the input features include a time and amount column, as well as 28 columns obtained through PCA. The output feature, labeled as "Class," indicates whether each record or transaction indicates the occurrence of credit card fraud.

## Data PreProcessing

To ensure that the input features were on the same scale, the **StandardScaler** method from the sklearn library was used to standardize the dataset prior to applying any other techniques. Once standardized, the dataset was split into training and testing sets, with a test size of 0.3.

## Balancing the Dataset

The original dataset was imbalanced, with a majority of records belonging to class 1 and only a few records belonging to class 0. As a result, the training dataset was also imbalanced, with only **6 records** of class 0 and 534 records of class 1. To address this issue, we used the **Synthetic Minority Over-sampling Technique (SMOTE)**, an algorithm designed to oversample the minority class and balance the dataset.

SMOTE works by generating synthetic examples of the minority class based on the existing samples, which are then added to the training dataset. This approach helps to balance the distribution of the classes and improve the performance of machine learning models.

In our case, we used the SMOTE method from the imblearn library to balance the training dataset. After applying SMOTE, the training dataset now contains **534 records of both classes**, resulting in a balanced dataset. This balanced dataset can then be used to train machine learning models that are less likely to be biased towards the majority class.

## Selecting the Samples

After balancing the dataset, we selected five samples using five different sampling techniques: 
* Random Sampling
* Systematic Sampling
* Stratified Sampling
* Cluster Sampling
* Convenience Sampling. 

Each of these sampling techniques has its own strengths and weaknesses, and selecting the appropriate technique depends on the specific requirements and constraints of the study.

For Random Sampling, Systematic Sampling, and Convenience Sampling, we calculated the sample sizes based on the formula: **Z^2p(1-p)/E^2**, where Z represents the confidence level, p represents the proportion of the population, and E represents the desired margin of error.

For Stratified Sampling, we used the formula: **Z^2p(1-p)/(E/S)^2**, where S represents the number of strata or subgroups in the population. Stratified Sampling is particularly useful when the population is heterogeneous and there are significant differences among subgroups.

For Cluster Sampling, we used the formula: **Z^2p(1-p)/(E/C)^2**, where C represents the number of clusters. Cluster Sampling is useful when the population is geographically dispersed and it is not feasible to sample individuals randomly from the entire population.

## Evaluation of Machine Learning Models on Different Samples

After obtaining the samples using various sampling techniques, we applied several machine learning models to each sample. The purpose of this was to assess how the different sampling techniques impact the performance of the machine learning models.

The machine learning models used are :
* Random Forest
* Decision Tree
* Gaussian Naive Bayes
* Logistic Regression
* Ridge Classifier

Each of these models has its own strengths and weaknesses, and selecting the appropriate model depends on the specific problem and data.

## Result 
The following are the accuracies achieved by applying different machine learning models to the different samples:

|                          | Random Sampling     | Systematic Sampling | Stratified Sampling     | Cluster Sampling     | Convenience Sampling |
| ------------------------ | ------------------- | ------------------- | ----------------------- | -------------------- | -------------------- |
| **Random Forest**        | 0\.987069 | 0\.982759 | **0\.991379** | 0\.987069  | 0\.987069  |
| **Decision Tree**        | 0\.956896 | 0\.939655 | 0\.969828     | 0\.939655  | 0\.948276  |
| **Gaussian Naive Bayes** | 0\.862068 | 0\.883620 | 0\.836207     | 0\.982759  | 0\.849138  |
| **Logistic Regression**  | 0\.862069 | 0\.870690 | 0\.875000     | 0\.586207  | 0\.870690  |
| **Ridge Classifier**     | 0\.836207 | 0\.844828 | 0\.844828     | 0\.478448 | 0\.823276  |

The **Random Forest Classifier** achieved the highest accuracy of **0.991379**, when applied to the sample obtained through stratified sampling. This finding indicates that stratified sampling could be a suitable sampling technique for this particular dataset and problem. 

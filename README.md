# The harm of SMOTE and other imbalance adjustments: a case study
 A case study to class imbalance solutions and the performance of medical prediction models

Author of this file and of the final version of the code: Ruben Willem van den Goorbergh

File last modified: January 2021

Code last modified: January 2021

This is code used for a case study regarding the effect of different approaches to deal with class imbalance on the performance of medical prediction models. This study will be extended with a simulation study in the first semester of 2021.

For questions, email:

r.vandengoorbergh@gmail.com

## Index
1. [Get started](#start)
2. [Variables](#variables)
3. [Imbalance solutions](#imbalance)
4. [Models](#models)
5. [Performance statistics](#performance)
6. [Calibration plots](#calibration)
7. [Research report](#report)

#### 1. Get started <a name="start"></a>
 a. All simulations were carried out in R 3.6.2<br /> 
 b. The .R files contain elaborate comments what happens in which part of the code, therefore only a description on where to find which process can be found in this file. <br />
 c. Note that in line 21 the working directory should be specified as the path to the folder "research_report_repository".<br />  
 d. The used R packages are: install.packages(c("tidyverse", "summarytools", "randomForest", "glmnet", "glmnetUtils", "xgboost", "data.table", "caret", "smotefamily", "xtable")) <br /> 
 e. Due to privacy concerns, the original data is not included in this repository. To illustrate where the data is supposed to be for the script in order to work  a proxy file is included. 
 f. The code can be found in the code folder

#### 2. Data & Variables <a name="variables"></a>
a. The data import and variable selection happens in lines 20 - 29<br />
b. The only cases where menoyn = 0(non-oncology centre patients) are included to create class imbalance<br />
c. Predictors included in the analysis are:<br />
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I. Age<br />
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;II. ovaryd1<br />
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;III. ovaryd3<br />
 These variables are included based on their statistical properties and not on clinical theory.<br /> 
 d. EDA and splitting train and test data happens in lines 31-64

 #### 3. Imbalance solutions <a name="imbalance"></a>
 a. Imbalance solutions are implemented in lines 66-99<br />
 b. Solutions used are:<br />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I. Random undersampling (RUS)<br />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;II. Random oversampling (ROS)<br />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;III. Synthetic Minority Oversampling TEchnique (SMOTE)
 
 #### 4. Models <a name="models"></a>
 a. Model fitting happens in lines 104 - 369<br />
 b. Models used are:<br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; I. Regular logistic regression<br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;II. Ridge logistic regression<br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;III. Random forest<br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;IIII. XGboost
 
 #### 5. Performance statistics <a name="performance"></a>
 a. Calculating the performance statistics (table 2 in the report) happens in lines 371 - 448.<br />
 b. The functions to calculate (except the c-statistic confidence intervals) can be found in Performance measure.R.
 
 #### 6. Calibration plots <a name="calibration"></a>
 a. Creating a grid of calibration plots happens in lines 491 - 585
 
 #### 7. Research report <a name="report"></a>
 a. The folder 'Research_report' contains a pdf of the final research report and a folder with all the files needed to compile the report.

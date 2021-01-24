# research_report_repo
 A case study to class imbalance solutions and the performance of medical prediction models

Author of this file and of the final version of the code: Ruben Willem van den Goorbergh

File last modified: January 2021

Code last modified: January 2021

This is code used for a case study regarding the effect of different approaches to deal with class imbalance on the performance of medical prediction models. This study will be extended with a simulation study in the first semester of 2021.



For questions, email:

r.vandengoorbergh@gmail.com

## How to...
1. [Get started](#start)
1. [Handle imbalance](#imbalance)

#### 1. Get started <a name="start"></a>
a. All simulations were carried out in R 3.6.2
b. The .R files contain elaborate comments what happens in which part of the code, therefore only a description on where to find which process can be found in this file.
c. Note that in line 21 the working directory should be specified as the path to the folder "research_report_repository".  
d. The used R packages are: install.packages(c("tidyverse", "summarytools", "randomForest", "glmnet", "glmnetUtils", "xgboost", "data.table", "caret", "smotefamily", "xtable")) 
e. Due to privacy concerns, the original data is not included in this repository. To illustrate where the data is supposed to be for the script in order to work a proxy file is included.

#### 1. Imbalance <a name="imbalance"></a>

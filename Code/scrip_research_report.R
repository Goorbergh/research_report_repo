#### Packages ####
#data clean
library(tidyverse) #1
library(summarytools)#1

#models
library(randomForest)#1
library(glmnet)#1
library(glmnetUtils)#1
library(xgboost) #1
library(data.table)

#imbalance
library(caret) #1
library(smotefamily) #1

# Performance
library(xtable) #1

#### Import data ####
setwd("/Users/ruben/GitHub/Thesis")
data <- read.delim('Data/iota_utrecht.txt', header = TRUE, )

# select only non-oncocenters (to create more imbalance)
data <- data %>% filter(menoyn != 1) 

# select variables used for model 
data2 <- data %>% 
  select(outcome1, Age, ovaryd1, ovaryd3) 

#### EDA ####
summary(data2)

dfSummary(data2)

# Check imbalance
data2 %>% 
  ggplot() +
  geom_bar(mapping = aes(outcome1))

n_minor <- data %>% 
  filter(outcome1 == 1) %>% 
  nrow()

n_major <- data %>% 
  filter(outcome1 == 0) %>% 
  nrow()

n_minor/n_major

#### train-test data ####
set.seed(133) # set seed for reproducibility 

# Sample indexes to select data for train set
train_index <- sample(x = nrow(data2), 
                      size = round(0.8 * length(data2$outcome1)), 
                      replace = FALSE)

# Split data into train and test set
train_set <- data2[train_index,] 
test_set <- data2[-train_index,]

# select outcome test set used in performance measures
test_outcome <- test_set$outcome1 

#### Class imbalance ####
# In this part different data sets are created using different class imbalance approaches

## SMOTE
train_smote <- SMOTE(X= train_set[,2:length(train_set)], target = train_set[,1])

train_smote <- train_smote$data

train_smote <- train_smote %>% 
  rename(outcome1 = class) 

train_smote$outcome1  <- as.factor(train_smote$outcome1)

train_smote %>% 
  ggplot()+
  geom_bar(mapping = aes(outcome1))


# Random oversampling

# creating separate predictors and outcome matrices to use in caret package functions
x <- train_set[,2:ncol(train_set)] 
y <- as.factor(train_set[,1])

train_up <- upSample(x = x, y = y, yname = 'outcome1')
train_up %>% 
  ggplot()+
  geom_bar(mapping = aes(outcome1))

# Random undersampling
train_down <- downSample(x = x, y = y, yname = 'outcome1')
train_down %>% 
  ggplot()+
  geom_bar(mapping = aes(outcome1))




#### Logistic regression ####
# In this part the logistic regression models are fitted.

## no adjustment##
# fit model
log_model <-glm(formula = outcome1~., family = 'binomial', data = train_set)
summary(log_model)
log_probs <- predict(log_model, test_set, type = 'response') # get probabilities
log_pred <- rep(0, length(log_probs)) # empty vector for class prediction
log_pred[log_probs > .5] = 1  # predict class with 0.5 threshold


## Oversampling ##
#fit model
log_model_up <-glm(formula = outcome1~., family = 'binomial', data = train_up)
summary(log_model_up)
log_probs_up <- predict(log_model_up, test_set, type = 'response') # get probabilities
log_pred_up <- rep(0, length(log_probs_up)) # empty vector for class prediction
log_pred_up[log_probs_up > .5] = 1 # predict class with 0.5 threshold



## Undersampling ##
# fit model
log_model_down <-glm(formula = outcome1~., family = 'binomial', data = train_down)
summary(log_model_down)
log_probs_down <- predict(log_model_down, test_set, type = 'response') # get probabilities
log_pred_down <- rep(0, length(log_probs_down)) # empty vector for class prediction
log_pred_down[log_probs_down > .5] = 1 # predict class with 0.5 threshold


## SMOTE ##
# Fit model
log_model_smote <-glm(formula = outcome1~., family = 'binomial', data = train_smote)
summary(log_model_smote)
log_probs_smote <- predict(log_model_smote, test_set, type = 'response') # get probabilities
log_pred_smote <- rep(0, length(log_probs_smote)) # empty vector for class prediction
log_pred_smote[log_probs_smote > .5] = 1 # predict class with 0.5 threshold


#### Ridge logistic regression ####
# In this part the ridge logistic regression models are fitted.

# Function to create a grid of 250 non 0 lambda values on a logarithmic scale
lseq <- function(from=0.001, to=64, length.out=251) {
  exp(seq(log(from), log(to), length.out = length.out))
}

lambdas <-  c(0, lseq()) # create grid by adding 0 to the 250 non-zero values for lambda

## no adjustment ##
x_train <- model.matrix(outcome1 ~., train_set)[,-1] # create matrix with predictors
y_train <- train_set$outcome1 # create vector with outcome

# Get hyper parameter
cv_out <- glmnet::cv.glmnet(x = x_train, y= y_train, alpha = 0, lambda = lambdas, 
                            family = 'binomial')

# Fit model
rid_model <- glmnet(x = x_train, y = y_train, alpha = 0, 
                    family = 'binomial',
                    lambda = cv_out$lambda.min)

x_test <- model.matrix(outcome1 ~., test_set)[,-1] # create matrix with predictors
y_test <- test_set$outcome1 # create vector with outcome
rid_probs <- predict(rid_model, newx = x_test, type = 'response') # get probabilities

rid_pred <- rep(0, length(rid_probs)) # vector for class prediction
rid_pred[rid_probs > .5] = 1  # predict classes with 0.5 threshold


## oversampling ##
x_train_up <- model.matrix(outcome1 ~., train_up)[,-1]# create matrix with predictors as model input
y_train_up <- train_up$outcome1 # create vector with outcome as model input

# Tune hyper parameter
cv_out_up <- glmnet::cv.glmnet(x = x_train_up, y= y_train_up, alpha = 0, lambda = lambdas, 
                               family = 'binomial')

# Fit model
rid_model_up <- glmnet(x = x_train_up, y = y_train_up, alpha = 0, 
                       family = 'binomial',
                       lambda = cv_out_up$lambda.min)

x_test <- model.matrix(outcome1 ~., test_set)[,-1]# create matrix with predictors for predict funtion
y_test <- test_set$outcome1 # create vector with outcome for predict function

rid_probs_up <- predict(rid_model_up, newx = x_test, type = 'response') # get probabilities

rid_pred_up <- rep(0, length(rid_probs_up)) # vector for class predictions
rid_pred_up[rid_probs_up > .5] = 1 # predict class with threshold 0.5


## under sampling ##
x_train_down <- model.matrix(outcome1 ~., train_down)[,-1]# create matrix with predictors as model input
y_train_down <- train_down$outcome1 # create vector with outcome as model input

# Tune hyper parameter
cv_out_down <- glmnet::cv.glmnet(x = x_train_down, y= y_train_down, alpha = 0, 
                                 lambda = lambdas, family = 'binomial')

# Fit model
rid_model_down <- glmnet(x = x_train_down, y = y_train_down, alpha = 0, 
                         family = 'binomial',
                         lambda = cv_out_down$lambda.min)

x_test <- model.matrix(outcome1 ~., test_set)[,-1] # create matrix with predictors for predict function
y_test <- test_set$outcome1 # create matrix with predictors for predict function
rid_probs_down <- predict(rid_model_down, newx = x_test, type = 'response') # get probabilities

rid_pred_down <- rep(0, length(rid_probs_down)) # Create vector for class predictions
rid_pred_down[rid_probs_down > .5] = 1 # predict classes with threshold 0.5

## SMOTE ##
x_train_smote <- model.matrix(outcome1 ~., train_smote)[,-1] # create matrix with predictors as model input
y_train_smote <- train_smote$outcome1 # create vector with outcome as model input

#Hyper parameter tuning
cv_out_smote <- glmnet::cv.glmnet(x = x_train_up, y= y_train_up, alpha = 0, 
                                  lambda = lambdas, family = 'binomial')

# Fit model
rid_model_smote <- glmnet(x = x_train_smote, y = y_train_smote, alpha = 0, 
                          family = 'binomial',
                          lambda = cv_out_smote$lambda.min)

x_test <- model.matrix(outcome1 ~., test_set)[,-1] # create matrix with predictors for predict function
y_test <- test_set$outcome1 # create vector with output for predict function

rid_probs_smote <- predict(rid_model_smote, newx = x_test, type = 'response')

rid_pred_smote <- rep(0, length(rid_probs_smote)) # Create vector for class prediction

#### Random Forrest ####
# In this part the random forest models are fitted.

###No adjustment ##
train_set$outcome1 <- as_factor(train_set$outcome1) # set outcome to factor to work for RF-function

# Fit model
rf_model <- randomForest(formula = outcome1~., data = train_set)
summary(rf_model)
rf_probs <- predict(rf_model, test_set, type = 'prob')
rf_probs <- rf_probs[,2]# select probabilities class == 1


rf_pred<- rep(0, length(rf_probs)) # create vector for class predictions
rf_pred [rf_probs > .5] = 1 # Predict classes with threshold 0.5

## Oversampling ##
train_up$outcome1 <- as_factor(train_up$outcome1) # set outcome to factor to work for RF-function

# Fit model 
rf_model_up <- randomForest(formula = outcome1~., data = train_up)
summary(rf_model_up)
rf_probs_up <- predict(rf_model_up, test_set, type = 'prob')
rf_probs_up <- rf_probs_up[,2] # select probabilities class == 1

rf_pred_up <- rep(0, length(rf_probs_up))# create vector for class predictions
rf_pred_up[rf_probs_up > .5] = 1 # Predict classes with threshold 0.5

## Undersampling ##
train_down$outcome1 <- as_factor(train_down$outcome1) # set outcome to factor to work for RF-function

# Fit model
rf_model_down <- randomForest(formula = outcome1~., data = train_down)
summary(rf_model_down)
rf_probs_down <- predict(rf_model_down, test_set, type = 'prob')
rf_probs_down <- rf_probs_down[,2] # select probabilities class == 1

rf_pred_down <- rep(0, length(rf_probs_down))# create vector for class predictions
rf_pred_down[rf_probs_down > .5] = 1 # Predict classes with threshold 0.5

## SMOTE ##
train_smote$outcome1 <- as_factor(train_smote$outcome1) # set outcome to factor to work for RF-function

rf_model_smote<- randomForest(formula = outcome1~., data = train_smote)
rf_probs_smote <- predict(rf_model_smote, test_set, type = 'prob')
rf_probs_smote <- rf_probs_smote[,2] # select probabilities class == 1

rf_pred_smote <- rep(0, length(rf_probs_smote))# create vector for class predictions
rf_pred_smote[rf_probs_smote > .5] = 1 # Predict classes with threshold 0.5

#### XGboost ####
# In this part the XGboost models are fitted.
#prepare data: traim_set_xg should contain only numeric values in order for the function to work
##No adjustment ##
train_set_xg <- train_set %>% 
  mutate(outcome1 = as.numeric(outcome1)-1)  # set outcome to numeric, see 1st line of xgboost part of the code

train_matrix <-  sparse.model.matrix(outcome1 ~., data = train_set_xg)[,-1] # create sparse matrix with predictors

output_vector <- train_set_xg$outcome1

dtrain <- xgb.DMatrix(data = train_matrix, label = output_vector) # create xgb.DMatrix object to train model

test_matrix <-  sparse.model.matrix(outcome1 ~., data = test_set)[,-1]

output_vector2 <- test_set$outcome1

dtest <- xgb.DMatrix(data = test_matrix, label = output_vector2) # create xgb.DMatrix object to test model

#fit model
bst <-  xgboost(data = dtrain, max.depth = 4, eta = 1, nrounds = 5,
                nthread = 2, objective = "binary:logistic")

bst_probs<- predict(bst, dtest) # get probabilities
bst_pred <- rep(0, length(bst_probs)) # create vector for class predictions
bst_pred[bst_probs > .5] = 1 # predict classes with threshold 0.5

## Oversampling ##
train_up_xg <- train_up %>% 
  mutate(outcome1 = as.numeric(outcome1)-1)  # set outcome to numeric, see 1st line of xgboost part of the code

train_matrix_up <-  sparse.model.matrix(outcome1 ~., data = train_up_xg)[,-1] # create sparse matrix with predictors

output_vector_up <- train_up_xg$outcome1

dtrain_up <- xgb.DMatrix(data = train_matrix_up, label = output_vector_up) # create xgb.DMatrix object to train model


#fit model
bst_up <-  xgboost(data = dtrain_up, max.depth = 4, eta = 1, nrounds = 5,
                nthread = 2, objective = "binary:logistic")

bst_probs_up<- predict(bst_up, dtest) # get probabilities
bst_pred_up <- rep(0, length(bst_probs_up)) # create vector for class predictions
bst_pred_up[bst_probs_up > .5] = 1 # predict classes with threshold 0.5

## undersampling ##
train_down_xg <- train_down %>% 
  mutate(outcome1 = as.numeric(outcome1)-1)  # set outcome to numeric, see 1st line of xgboost part of the code

train_matrix_down <-  sparse.model.matrix(outcome1 ~., data = train_down_xg)[,-1] # create sparse matrix with predictors

output_vector_down <- train_down_xg$outcome1

dtrain_down <- xgb.DMatrix(data = train_matrix_down, label = output_vector_down) # create xgb.DMatrix object to train model


#fit model
bst_down <-  xgboost(data = dtrain_down, max.depth = 4, eta = 1, nrounds = 5,
                   nthread = 2, objective = "binary:logistic")

bst_probs_down<- predict(bst_down, dtest) # get probabilities
bst_pred_down <- rep(0, length(bst_probs_down)) # create vector for class predictions
bst_pred_down[bst_probs_down > .5] = 1 # predict classes with threshold 0.5

## SMOTE ##
train_smote_xg <- train_smote %>% 
  mutate(outcome1 = as.numeric(outcome1)-1)  # set outcome to numeric, see 1st line of xgboost part of the code

train_matrix_smote <-  sparse.model.matrix(outcome1 ~., data = train_smote_xg)[,-1] # create sparse matrix with predictors

output_vector_smote <- train_smote_xg$outcome1

dtrain_smote <- xgb.DMatrix(data = train_matrix_smote, label = output_vector_smote) # create xgb.DMatrix object to train model


#fit model
bst_smote <-  xgboost(data = dtrain_smote, max.depth = 4, eta = 1, nrounds = 5,
                     nthread = 2, objective = "binary:logistic")

bst_probs_smote<- predict(bst_smote, dtest) # get probabilities
bst_pred_smote <- rep(0, length(bst_probs_smote)) # create vector for class predictions
bst_pred_smote[bst_probs_smote > .5] = 1 # predict classes with threshold 0.5

#### Paper statistics ####

# Creating table with all estimated probabilities to use in performance measure
# functions
probs_table <- cbind(log_probs, log_probs_up, log_probs_down, log_probs_smote,
                     rid_probs, rid_probs_up, rid_probs_down, rid_probs_smote,
                     (rf_probs + 0.001), (rf_probs_up + 0.001), 
                     (rf_probs_down + 0.001), (rf_probs_smote + 0.001),
                     bst_probs, bst_probs_up, bst_probs_down, bst_probs_smote)

# Creating table with all predicted classes to use in performance measure
# functions
pred_table <- cbind(log_pred, log_pred_up, log_pred_down, log_pred_smote,
                    rid_pred, rid_pred_up, rid_pred_down, rid_pred_smote,
                    (rf_pred), (rf_pred_up), (rf_pred_down), (rf_pred_smote),
                    bst_pred, bst_pred_up, bst_pred_down, bst_pred_smote)

# Source custom written performance measure functions
source('Performance measures.R')

# Creating empty vectors for performance measures
accuracy_vector <- rep(NA, ncol(pred_table))
sensitivity_vector <- rep(NA, ncol(pred_table))
specificity_vector <- rep(NA, ncol(pred_table))

# Loop over all models and imbalance approaches to get performance measures
for (i in 1:ncol(pred_table)){
  accuracy_vector[i] <- accuracy(pred_table[,i], test_outcome)
  sensitivity_vector[i] <- sensitivity(pred_table[,i], test_outcome)
  specificity_vector[i] <- specificity(pred_table[,i], test_outcome)
}

# Create matrix to store calibration measures (intercept & slope + CI)
calibration_matrix <- matrix(ncol = 6, nrow = ncol(probs_table))

# Calculate calibration measures for all models
for (i in 1:ncol(probs_table)) {
  calibration_matrix[i,] <- calibration(probs = probs_table[,i], outcome = test_outcome)
}

# Create matrix to store c statistics + CI
cstat_matrix <- matrix(nrow = ncol(probs_table), ncol = 3)

# Calculate c-statistic + CI
for (i in 1:ncol(probs_table)){
  mat <- cbind(probs_table[,i], test_outcome)
  x <- data.frame(mat) %>% 
    filter(test_outcome == 1)
  y <- data.frame(mat) %>% 
    filter(test_outcome == 0)
  cstat_matrix[i,] <- auRoc::auc.nonpara.mw(x = x[,1], 
                                   y = y[,1],
                                   method = 'pepe')
}

# Bind all results together
results <- cbind(accuracy_vector, sensitivity_vector, specificity_vector,
                 cstat_matrix, calibration_matrix, eci)

# Round results to 2 digits
results <- format(round(results, digits = 2), nsmall = 2)

# Name rows and columns of the results object
colnames(results) <- c("Accuracy", "Sensitivity", "Specificity", "C-statistic",
                       "C-statistic lower", "C-statistic upper", "CIL", 
                       "Lower CIL", "Upper CIL", "Calibration slope", 
                       "Lower slope", "Upper slope", "ECI")
rownames(results) <- c("LR", "LR up", "LR down", "LR smote", 
                       "Ridge", "Ridge up", "Ridge down", "Ridge smote",
                       "RF", "RF up", "RF down", "RF smote", 
                       "Bst", "Bst up", "Bst down", "Bst smote")


results <- data.frame(results) %>% 
  apply(2, as.character) %>% 
  apply(2, as.numeric)

results <- data.frame(results)


# Loop over results object to get all CI's in parentheses
for (i in 1:nrow(results)){
  results$C.statistic[i] <- str_c(results$C.statistic[i],
                   " (", 
                   results$C.statistic.lower[i],
                   " — ",
                   results$C.statistic.upper[i],
                   ")")
  results$CIL[i] <- str_c(results$CIL[i],
                               " (", 
                               results$Lower.CIL[i],
                               " — ",
                               results$Upper.CIL[i],
                               ")")
  results$Calibration.slope[i] <- str_c(results$Calibration.slope[i],
                               " (", 
                               results$Lower.slope[i],
                               " — ",
                               results$Upper.slope[i],
                               ")")
}


# Remove old CI measure columns
results <- results %>% 
  select(!c(C.statistic.lower, C.statistic.upper, 
            Upper.CIL, Lower.CIL, 
            Upper.slope, Lower.slope))

# Get LaTeX table
xtable(results)

## Calibration plots ##

# Store probabilities as numeric in order to fit loes-curve
rid_probs <-  as.numeric(rid_probs)
rid_probs_up <- as.numeric(rid_probs_up)
rid_probs_down <- as.numeric(rid_probs_down)
rid_probs_smote <- as.numeric(rid_probs_smote)

# Create dataframe with results as input for ggplot
cal_plot_df <- data.frame(cbind(log_probs, log_probs_up, log_probs_down, log_probs_smote,
                     as.numeric(rid_probs), as.numeric(rid_probs_down), 
                     as.numeric(rid_probs_up), as.numeric(rid_probs_smote),
                     rf_probs, rf_probs_up, rf_probs_up, rf_probs_smote,
                     bst_probs, bst_probs_up, bst_probs_down, bst_probs_smote))



df_probs <- rbind(as_tibble(cbind(log_probs, test_outcome)) %>% 
    mutate(model = 'LR') %>%
    mutate(imbalance_solution = 'None') %>% 
      rename(probability = log_probs),
    as_tibble(cbind(rid_probs, test_outcome)) %>% 
       mutate(model = 'LR_rid') %>%
       mutate(imbalance_solution = 'None') %>% 
      rename(probability = rid_probs),
    as_tibble(cbind(rf_probs, test_outcome)) %>% 
       mutate(model = 'RF') %>% 
       mutate(imbalance_solution = 'None') %>% 
      rename(probability = rf_probs),
    as_tibble(cbind(bst_probs, test_outcome)) %>% 
       mutate(model = 'Bst') %>%
       mutate(imbalance_solution = 'None') %>% 
      rename(probability = bst_probs))

df_probs_up <- rbind(as_tibble(cbind(log_probs_up, test_outcome)) %>% 
                    mutate(model = 'LR') %>%
                    mutate(imbalance_solution = 'ROS') %>% 
                    rename(probability = log_probs_up),
                  as_tibble(cbind(rid_probs_up, test_outcome)) %>% 
                    mutate(model = 'LR_rid') %>%
                    mutate(imbalance_solution = 'ROS') %>% 
                    rename(probability = rid_probs_up),
                  as_tibble(cbind(rf_probs_up, test_outcome)) %>% 
                    mutate(model = 'RF') %>% 
                    mutate(imbalance_solution = 'ROS') %>% 
                    rename(probability = rf_probs_up),
                  as_tibble(cbind(bst_probs_up, test_outcome)) %>% 
                    mutate(model = 'Bst') %>%
                    mutate(imbalance_solution = 'ROS') %>% 
                    rename(probability = bst_probs_up))

df_probs_down <- rbind(as_tibble(cbind(log_probs_down, test_outcome)) %>% 
                       mutate(model = 'LR') %>%
                       mutate(imbalance_solution = 'RUS') %>% 
                       rename(probability = log_probs_down),
                     as_tibble(cbind(rid_probs_down, test_outcome)) %>% 
                       mutate(model = 'LR_rid') %>%
                       mutate(imbalance_solution = 'RUS') %>% 
                       rename(probability = rid_probs_down),
                     as_tibble(cbind(rf_probs_down, test_outcome)) %>% 
                       mutate(model = 'RF') %>% 
                       mutate(imbalance_solution = 'RUS') %>% 
                       rename(probability = rf_probs_down),
                     as_tibble(cbind(bst_probs_down, test_outcome)) %>% 
                       mutate(model = 'Bst') %>%
                       mutate(imbalance_solution = 'RUS') %>% 
                       rename(probability = bst_probs_down))

df_probs_smote <- rbind(as_tibble(cbind(log_probs_smote, test_outcome)) %>% 
                         mutate(model = 'LR') %>%
                         mutate(imbalance_solution = 'SMOTE') %>% 
                         rename(probability = log_probs_smote),
                       as_tibble(cbind(rid_probs_smote, test_outcome)) %>% 
                         mutate(model = 'LR_rid') %>%
                         mutate(imbalance_solution = 'SMOTE') %>% 
                         rename(probability = rid_probs_smote),
                       as_tibble(cbind(rf_probs_smote, test_outcome)) %>% 
                         mutate(model = 'RF') %>% 
                         mutate(imbalance_solution = 'SMOTE') %>% 
                         rename(probability = rf_probs_smote),
                       as_tibble(cbind(bst_probs_smote, test_outcome)) %>% 
                         mutate(model = 'Bst') %>%
                         mutate(imbalance_solution = 'SMOTE') %>% 
                         rename(probability = bst_probs_smote))

df <- rbind(df_probs, df_probs_up, df_probs_down, df_probs_smote) %>% 
  rename(event = test_outcome)

# Create grid of calibration plots with all models on y-axis and imbalance 
# solutions on the x-axis
cal_plot <- df %>%
              ggplot(aes(x = probability, y = event)) +
              ylab('Event fraction') +
              scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
              scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
              geom_smooth(color = "red", se = TRUE,  method = "loess") +
              geom_abline() +
              scale_colour_manual(name="legend", values=c("black", "red")) +
              facet_grid(rows = vars(model), cols = vars(imbalance_solution)) +
              theme_minimal() 
             
# Save calibration plot
ggsave("calibration_plots.jpeg", plot = cal_plot)


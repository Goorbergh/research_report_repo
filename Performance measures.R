# Evaluation functions

#### Discrimination ####


# accuracy
accuracy <- function(pred, outcome){
  correct_predictions <- sum(pred == outcome)
  total_predictions <- length(pred)
  correct_predictions / total_predictions
}

# sensitivity 
sensitivity <- function(pred, outcome) {
  true_positives <- sum(pred == 1 & outcome == 1)
  false_negatives <-  sum(pred == 0 & outcome == 1)
  true_positives/(true_positives+false_negatives)
}

# specificity
specificity <- function(pred, outcome) {
  true_negatives <- sum(pred == 0 & outcome == 0)
  false_positives <-  sum(pred == 1 & outcome == 0)
  true_negatives/(true_negatives+false_positives)
}

# c-statistic
c.stat2 <- function(probs, outcome){
  probs <- as.matrix(probs)
  cats <- sort(unique(outcome))
  n_cat <- length(cats)
  n0   <- sum(outcome == cats[2])
  n1   <- length(outcome) - n0
  r <- rank(probs[,1])
  S0 <- sum(as.numeric(r[outcome == cats[2]]))
  (S0 - n0 * (n0 + 1)/2)/(as.numeric(n0) * as.numeric(n1))
}

# with CI
auc <- matrix(nrow = ncol(probs_table), ncol = 3)

for (i in 1:ncol(probs_table)){
  mat <- cbind(probs_table[,i], test_outcome)
  x <- data.frame(mat) %>% 
    filter(test_outcome == 1)
  y <- data.frame(mat) %>% 
    filter(test_outcome == 0)
  auc[i,] <- auRoc::auc.nonpara.mw(x = x[,1], 
                                   y = y[,1],
                                   method = 'pepe')
}

#### Calibration ####


# Calibration curves + intercepts
calibration <- function(probs,outcome){
  slope_model <- glm(outcome ~ log(probs/(1-probs)), family = "binomial")
  slope <- coef(slope_model)[2]
  slope_ci <- confint(slope_model)[2,]
  intercept_model <- glm(test_outcome ~ 1, 
                         offset = log(probs/(1-probs)), 
                         family = "binomial")
  intercept <- coef(intercept_model)
  intercept_ci <-  confint(intercept_model)
  return(c(intercept, intercept_ci, slope, slope_ci))
}

# ECI
eci_bvc <- function(calout,preds,k){
  (mean((preds-fitted(calout))*(preds-fitted(calout))))*(100*k/2)
}

calout_f <- function(outcome, probs){ 
  loess(outcome ~ log(probs/(1-probs)))
}

calout <- list()
  
for (i in 1:ncol(probs_table)){
  calout[[i]] <- calout_f(test_outcome, probs = probs_table[,i])
}

eci <- rep(NA, ncol(probs_table))

for (i in 1:length(calout)) {
  eci[i] <- eci_bvc(calout = calout[[i]], probs_table[,i], k = 2)
}

eci



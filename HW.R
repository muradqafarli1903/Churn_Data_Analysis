library(tidyverse)
library(skimr)
library(inspectdf)
library(caret)
library(glue)
library(highcharter)
library(h2o)
install.packages("scorecard")
library(scorecard)
setwd('Churn_Modelling (1).csv')
raw <- read.csv('C:/Users/user/Masaüstü/all/Data science/R/R week 5/Churn_Modelling (1).csv')
df %>% glimpse()
df$Exited %>% table() %>% prop.table()
names(raw) <- names(raw) %>% 
  str_replace_all(" ","_") %>%
  str_replace_all("-","_") %>%
  str_replace_all("\\(","_") %>% 
  str_replace_all("\\)","") %>%
  str_replace_all("\\<=","LESS.EQUAL") %>%
  str_replace_all("\\>=","MORE.EQUAL") %>%
  str_replace_all("\\<","LESS") %>%
  str_replace_all("\\>","MORE") %>%
  str_replace_all("\\/","_") %>% 
  str_replace_all("\\:","_") %>% 
  str_replace_all("\\.","_") %>% 
  str_replace_all("\\,","_")
iv <- raw %>% iv(y='Exited') %>% as.tibble() %>% mutate(info_value = round(info_value, 3)) %>%
  arrange(desc(info_value))
iv
ivars <- iv %>% 
  filter(info_value>0.02) %>% 
  select(variable) %>% .[[1]]
ivars
raw.iv <- raw %>% select(Exited,ivars)

raw.iv %>% dim()

raw.iv <- raw.iv %>% select(-Surname)
bins <- raw.iv %>% woebin("Exited")
dt_list <- raw.iv %>% 
  split_df("Exited", ratio = 0.8, seed = 123)

train_woe <- dt_list$train %>% woebin_ply(bins) 
test_woe <- dt_list$test %>% woebin_ply(bins)

names <- train_woe %>% names() %>% gsub("_woe","",.)                   
names(train_woe) <- names              ; names(test_woe) <- names
train_woe %>% inspect_na() %>% tail(2) ; test_woe %>% inspect_na() %>% tail(2)
#----------------------------VIF-----------------------------------------#
  
target <- 'Exited'
features <- train_woe %>% select(-Exited) %>% names()

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")
glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features %in% coef_na]
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")

# VIF (Variance Inflation Factor) 
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")

while(glm %>% vif() %>% arrange(desc(gvif)) %>% .[1,2] >= 1.5){
  afterVIF <- glm %>% vif() %>% arrange(desc(gvif)) %>% .[-1,"variable"]
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = train_woe, family = "binomial")
}

glm %>% vif() %>% arrange(desc(gvif)) %>% pull(variable) -> features 
h2o.init()

train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
test_h2o <- test_woe %>% select(target,features) %>% as.h2o()

model <- h2o.glm(
  x = features, y = target, family = "binomial", 
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T)

while(model@model$coefficients_table %>%
      as.data.frame() %>%
      select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] >= 0.05){
  model@model$coefficients_table %>%
    as.data.frame() %>%
    select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
  test_h2o <- test_woe %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target, family = "binomial", 
    training_frame = train_h2o, validation_frame = test_h2o,
    nfolds = 10, seed = 123, remove_collinear_columns = T,
    balance_classes = T, lambda = 0, compute_p_values = T)
}
model@model$coefficients_table %>%
  as.data.frame() %>%
  select(names,p_value) %>%
  mutate(p_value = round(p_value,3))

model@model$coefficients %>%
  as.data.frame() %>%
  mutate(names = rownames(model@model$coefficients %>% as.data.frame())) %>%
  `colnames<-`(c('coefficients','names')) %>%
  select(names,coefficients)

h2o.varimp(model) %>% as.data.frame() %>% .[.$percentage != 0,] %>%
  select(variable, percentage) %>%
  hchart("pie", hcaes(x = variable, y = percentage)) %>%
  hc_colors(colors = 'orange') %>%
  hc_xAxis(visible=T) %>%
  hc_yAxis(visible=T)


# ---------------------------- Evaluation Metrices ----------------------------

# Prediction & Confision Matrice
pred <- model %>% h2o.predict(newdata = test_h2o) %>% 
  as.data.frame() %>% select(p1,predict)

model %>% h2o.performance(newdata = test_h2o) %>%
  h2o.find_threshold_by_max_metric('f1')

eva <- perf_eva(
  pred = pred %>% pull(p1),
  label = dt_list$test$creditability %>% as.character() %>% as.numeric(),
  binomial_metric = c("auc","gini"),
  show_plot = "roc")

eva$confusion_matrix$dat

# Check overfitting ----
model %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)


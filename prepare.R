##
library(tidyverse)
library(xgboost)
library(rsample)
library(yardstick)
library(data.table)
library(caret)
library(recipes)
library(dtplyr)

## read data
train <- fread("data/train.csv", stringsAsFactors = F)


## drop ID column
train <- train[,-1]

## drop variables which have only one value
val_one <-
  map_dbl(train, ~ .x %>% uniqueN) ==  1

train <- train[,!val_one, with = F]

## drop categorical variavles which have too many values
# this also drops date variables!
val_many <-
  map_dbl(train, ~ .x %>% uniqueN) > 10 &
  map_chr(train, typeof) ==   "character"

train <- train[,!val_many, with = F]

## drop inbalanced variables
rec <-
  recipe(target ~ .,  train) %>%
  step_nzv(all_nominal(),
           -all_outcomes(),
           options = list(freq_cut = 95 / 5, unique_cut = 10)) %>%
  prep(train)

train <- bake(rec, train)
train <- train %>%
  mutate_if(is.factor, as.character)
## split dataset
train_valid_sp <- initial_split(train, prop = 0.8)
train <- training(train_valid_sp)
valid <- testing(train_valid_sp)

rm(train_valid_sp)

## batch execution of variable clenzing
last_cl <- length(train)

for (ii in split(1:(length(train) - 1),
                 ceiling((1:(length(
                   train
                 ) - 1)) / 5))) {
  if (any(map_lgl(train[, c(ii, last_cl)], is.character))) {
    rec <- recipe(target ~ ., data = train[, c(ii, last_cl)]) %>%
      step_meanimpute(all_numeric(),-all_outcomes()) %>%
      step_modeimpute(all_nominal(),-all_outcomes()) %>%
      step_center(all_numeric(),-all_outcomes()) %>%
      step_scale(all_numeric(),-all_outcomes()) %>%
      step_dummy(all_nominal(),-all_outcomes()) %>%
      prep(train[, c(ii, last_cl)])
  } else {
    rec <- recipe(target ~ ., data = train[, c(ii, last_cl)]) %>%
      step_meanimpute(all_numeric(),-all_outcomes()) %>%
      step_center(all_numeric(),-all_outcomes()) %>%
      step_scale(all_numeric(),-all_outcomes()) %>%
      prep(train[, c(ii, last_cl)])
  }
  
  # Predictors
  x_train_tmp <-
    bake(rec, newdata = train[, c(ii, last_cl)]) %>% dplyr::select(-target)
  x_valid_tmp  <-
    bake(rec, newdata = valid[, c(ii, last_cl)]) %>% dplyr::select(-target)
  
  if (ii[1] == 1) {
    x_train <- x_train_tmp
    x_valid <- x_valid_tmp
  } else {
    x_train <- cbind(x_train, x_train_tmp)
    x_valid <- cbind(x_valid, x_valid_tmp)
  }
  message(ii[1])
}

rm(x_train_tmp, x_valid_tmp, rec)

x_train <- x_train %>% as_tibble()
x_valid <- x_valid %>% as_tibble()

## check NA
is_na <- function(x)
  sum(is.na(x)) > 0
is_nan <- function(x)
  sum(is.nan(x)) > 0

var_na <- map_lgl(x_train, is_na)
x_train <-  x_train[,!var_na]
var_nan <- map_lgl(x_train, is_nan)
x_train <-  x_train[,!var_nan]



## check linear dependence
colnames(x_train) %>%
  stringr::str_count("_") %>%
  {
    . > 1
  } %>%
  x_train[, .] %>%
  data.matrix() %>%
  findLinearCombos() -> var_lc_dummy

x_train <- x_train[,-var_lc_dummy$remove]


##
batch_size <- 2000
try_num <- 10

lc_cl <- list()

for (ii in seq_len(try_num)) {
  batch_sample <- x_train %>% 
    sample_n(batch_size) %>% 
    data.matrix() %>%
    findLinearCombos() -> var_lc
  lc_cl <- splice(lc_cl, var_lc$)
  message(ii)
  }

x_train[, reduce(lc_cl, intersect)] %>% 
  data.matrix() %>%
  findLinearCombos() -> var_lc_sum


x_train <- x_train[, -var_lc_sum$remove]
x_valid <- x_valid[, colnames(x_train)]

## Response variables for training and testing sets
y_train <- train$target
y_valid  <- valid$target


##
saveRDS(x_train, "data/intermidiate/x_train.rds")
saveRDS(x_valid, "data/intermidiate/x_valid.rds")
saveRDS(y_train, "data/intermidiate/y_train.rds")
saveRDS(y_valid, "data/intermidiate/y_valid.rds")

beepr::beep()

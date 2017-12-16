library(speedglm)

                 
system.time({
  lg <- speedglm(target ~ .,
                 mutate(x_train, target = y_train) %>% 
                   sample_frac(0.3),
                 family = binomial(link = "logit"),
                 set.default = list(tol.solve = 1e-20),
                 method = "Cholesky")
  
})


pred <- predict(lg, newdata = mutate(x_valid, target = y_valid))

tibble(
  truth = y_valid %>% 
    factor(levels = 1:0),
  y_pred = ifelse(pred > 0.5, 1, 0)
  
) %>% 
  roc_auc(truth, y_pred)

                
                
submission <- data.frame(ID=test$ID)
submission$target <- NA 
for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) {
  submission[rows, "target"] <- predict(lg, test[rows,])
}

cat("saving the submission file\n")
write_csv(submission, "logistic_submission.csv")

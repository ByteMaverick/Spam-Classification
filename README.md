
# Implementing Logistic Regression model for Spam Classification(Report)

### Datasets
The dataset consists of word frequency features extracted from spam emails. The dataset is highly  
imbalanced, meaning that there are significantly more ham than spam emails. The columns have the  
common words found in spam emails, and the rows are their frequency.  
Train set: 4459 rows and 1365 columns  
Test Set: 1115 rows and 1365 columns  

### Model Development
I implemented logistic regression by following the five-step process you provided and used my  
implementation of linear regression for reference.  

#### Problem 1:
I kept getting an error in the lines where I was calculating the train and validation loss(fit method).  
The error stated that log(0) is undefined. At first, I didn’t understand why this issue was occurring.  
However, after some debugging, I noticed that there were instances in `y_train_pred` and `y_val_pred`  
where the values were zero.  
This was the issue since `y_train_pred` and `y_val_pred` are used to calculate the loss, and in the loss  
function, `log(y_hat)` is computed. Since log(0) is undefined, it caused the error.  
To fix this issue, I had to use an epsilon (a very small value) wherever `y_train_pred` and `y_val_pred`  
were 0, ensuring that no zero values were passed into the logarithm function.  

#### Problem 2:
I had a bit of an issue deciding how to calculate the train and the value loss.  
In the end, I decided to store the loss from the last epoch for each fold. After the 5 folds, I calculated  
the mean of losses and used them as the training and validation loss respectively.  

### What can be improved(Outside the scope of the assignment)?

- Use Mini-Batch GD instead of SGD for faster convergence (Less training time)  
- Implementing a Data Flow pipeline, rather than storing the entire dataset in memory, we can  
  upload in batches to improve efficiency (I have used this technique in deep learning models,  
  not sure how I’ll implement this in traditional ML).  

### Model Results

The best model parameters are 200 epochs with a learning rate of 0.001. I arrived at these parameters  
based on the best loss across all tested values and also used a threshold. The threshold helped me  
prioritize a model with low loss and a small difference between training and validation loss, to avoid  
overfitting. Without the threshold condition the best model parameters were 200 epochs with a  
learning rate of 0.01 (the model was slightly overfitting).  
I preferred a model that generalizes well, even if it has a slightly higher validation loss, over a model  
that poorly generalizes but has a very low validation loss.  

### Model Performance:
![Alt Text](/Spam-Classification/img.png)

- Accuracy: 95.605381  
- F1 Score: 0.854599  
- Test Loss: 0.149459  

Overall, the model performed well and the graph suggests the same (validation loss is slightly higher  
than training loss), but if I want to use the model with lower validation loss with poor generalization. I  
can try the following things:

- Implement early stopping  
- Use L1/L2 regularization to penalize the model when it is overfitting.  

## Light GBM model vs XGBoost Model
[blog](https://pyligent.github.io/2019-03-30-lightGBM_XGBoost/)

[XGBoost Model - Housing Prices](https://github.com/Pyligent/lightGBM_vs_XGBoost/blob/master/XGBoost%20and%20Gradient%20Boosting.ipynb)
### 1.Light GBM model

- **Light GBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithm, used for ranking, classification and many other machine learning tasks.**
- [Reference Link](https://lightgbm.readthedocs.io/en/latest/), [GitHub](https://github.com/Microsoft/LightGBM) 
- **Key Features:**
  - **Many boosting tools use pre-sort-based algorithms(e.g. default algorithm in xgboost) for decision tree learning. It is a simple solution, but not easy to optimize.**
  - **LightGBM uses histogram-based algorithms, which bucket continuous feature (attribute) values into discrete bins. This speeds up training and reduces memory usage. Advantages of histogram-based algorithms include the following:**
    - Reduced cost of calculating the gain for each split
    - Use histogram subtraction for further speedup
    - Reduce memory usage
    - Reduce communication cost for parallel learning
  - **LightGBM grows trees leaf-wise (best-first). It will choose the leaf with max delta loss to grow. Holding #leaf fixed, leaf-wise algorithms tend to achieve lower loss than level-wise algorithms.**
  - **LightGBM supports the following applications:**
      - regression, the objective function is L2 loss
      - binary classification, the objective function is logloss
      - multi classification
      - cross-entropy, the objective function is logloss and supports training on non-binary labels
      - lambdarank, the objective function is lambdarank with NDCG   
      
### Tuning Light GBM parameters   
- Light GBM uses leaf wise splitting over depth wise splitting which enables it to converge much faster but also leads to overfitting. So here is a quick guide to tune the parameters in Light GBM.

- For best fit
    - **num_leaves:** Light GBM model is to split leaf-wise nodes rather than depth-wise. Hence num_leaves set must be smaller than 2^(max_depth) otherwise it may lead to overfitting. Light GBM does not have a direct relation between num_leaves and max_depth and hence the two must not be linked with each other.
    - **min_data_in_leaf:** Key parameter to avoid overfitting. Setting its value smaller may cause overfitting and hence must be set accordingly. Its value should be hundreds to thousands of large datasets.
    - **max_depth:** It specifies the maximum depth or level up to which tree can grow.
 

- For faster speed
    - **bagging_fraction:** Is used to perform bagging for faster results
    - **feature_fraction:** Set fraction of the features to be used at each iteration
    - **max_bin:** Smaller value of max_bin can save much time as it buckets the feature values in discrete bins which is computationally inexpensive.
 

- For better accuracy
    - **more training data**
    - **num_leaves:** Setting it to high value produces deeper trees with increased accuracy but lead to overfitting. Hence its higher value is not preferred.
    - **max_bin:** Setting it to high values has similar effect as caused by increasing value of num_leaves and also slower our training procedure

### 2. XGBoost Model

- **XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data.**
- [Reference Link](https://xgboost.readthedocs.io/en/latest/), [GitHub](https://github.com/dmlc/xgboost),[Tutorials](https://github.com/dmlc/xgboost/tree/master/demo#tutorials)
- **Key Features:**
    - **Gradient Boosting algorithm also called gradient boosting machine including the learning rate.**
    - **Stochastic Gradient Boosting with sub-sampling at the row, column and column per split levels.**
    - **Regularized Gradient Boosting with both L1 and L2 regularization.**
    - **Sparse Aware - Handling missing values** implementation with automatic handling of missing data values.
    - **Block Structure** to support the parallelization of tree construction.
    - **Continued Training** so that you can further boost an already fitted model on new data.
    - **Tree Pruning:** A GBM would stop splitting a node when it encounters a negative loss in the split. Thus it is more of a greedy algorithm. XGBoost on the other hand make splits upto the max_depth specified and then start pruning the tree backwards and remove splits beyond which there is no positive gain.
    - **Built-in Cross-Validation**  XGBoost allows user to run a cross-validation at each iteration of the boosting process and thus it is easy to get the exact optimum number of boosting iterations in a single run.
    

### Tuning XGBoost parameters

- ***General Parameters***
    - **booster [default=gbtree]**   
        Select the type of model to run at each iteration. It has 2 options:
            gbtree: tree-based models
            gblinear: linear models
    - **silent [default=0]:**   
        Silent mode is activated is set to 1, i.e. no running messages will be printed.It’s generally good to keep it 0 as the messages might help in understanding the model.
    - **nthread [default to maximum number of threads available if not set]**   
        This is used for parallel processing and number of cores in the system should be entered.If you wish to run on all cores, value should not be entered and algorithm will detect automatically
 

- **Booster Parameters**

    - **eta [default=0.3]**   
        Analogous to learning rate in GBM. Makes the model more robust by shrinking the weights on each step.Typical final values to be used: 0.01-0.2
    - **min_child_weight [default=1]**   
        Defines the minimum sum of weights of all observations required in a child. This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.Too high values can lead to under-fitting hence, it should be tuned using CV.
    - **max_depth [default=6]**   
        The maximum depth of a tree, same as GBM.Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.Should be tuned using CV.Typical values: 3-10
    - **max_leaf_nodes**   
        The maximum number of terminal nodes or leaves in a tree.Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.If this is defined, GBM will ignore max_depth.
    - **gamma [default=0]**   
        A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
    - **max_delta_step [default=0]**   
        In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative.Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
    - **subsample [default=1]**   
        Same as the subsample of GBM. Denotes the fraction of observations to be randomly samples for each tree.Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting. Typical values: 0.5-1
    - **colsample_bytree [default=1]**   
        Similar to max_features in GBM. Denotes the fraction of columns to be randomly samples for each tree.Typical values: 0.5-1
    - **colsample_bylevel [default=1]**   
        Denotes the subsample ratio of columns for each split, in each level.
    - **lambda [default=1]**   
        L2 regularization term on weights (analogous to Ridge regression) This used to handle the regularization part of XGBoost. Though many data scientists don’t use it often, it should be explored to reduce overfitting.
    - **alpha [default=0]**   
        L1 regularization term on weight (analogous to Lasso regression). Can be used in case of very high dimensionality so that the algorithm runs faster when implemented
    - **scale_pos_weight [default=1]**   
        A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.
 

- **Learning Task Parameters**
These parameters are used to define the optimization objective the metric to be calculated at each step.

    - **objective [default=reg:linear]**   
        This defines the loss function to be minimized. Mostly used values are:
            binary:logistic –logistic regression for binary classification, returns predicted probability (not class)
            multi:softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities)
        you also need to set an additional num_class (number of classes) parameter defining the number of unique classes. multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class.
    - **eval_metric [ default according to objective ]**   
        The metric to be used for validation data.The default values are rmse for regression and error for classification.
        Typical values are:
            **rmse** – root mean square error
            **mae** – mean absolute error
            **logloss** – negative log-likelihood
            **error** – Binary classification error rate (0.5 threshold)
            **merror** – Multiclass classification error rate
            **mlogloss** – Multiclass logloss
            **auc**: Area under the curve
    - **seed [default=0]**   
        The random number seed.Can be used for generating reproducible results and also for parameter tuning.

    

    
### 3. "Census Income" Dataset to test Light GBM and XGBoost model   

![Model Performence](https://github.com/Pyligent/lightGBM_vs_XGBoost/blob/master/gbm_result.png)   
![light GBM ROC Curve](https://github.com/Pyligent/lightGBM_vs_XGBoost/blob/master/lightgbm.png)
![XGBoost ROC Curve](https://github.com/Pyligent/lightGBM_vs_XGBoost/blob/master/xgboost.png)








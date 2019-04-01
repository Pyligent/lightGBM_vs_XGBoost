## Light GBM model vs XGBoost Model

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

### 2. XGBoost Model

- **XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data.**
- [Reference Link](https://xgboost.readthedocs.io/en/latest/), [GitHub](https://github.com/dmlc/xgboost),[Tutorials](https://github.com/dmlc/xgboost/tree/master/demo#tutorials)
- **Key Features:**
    - **Gradient Boosting algorithm also called gradient boosting machine including the learning rate.**
    - **Stochastic Gradient Boosting with sub-sampling at the row, column and column per split levels.**
    - **Regularized Gradient Boosting with both L1 and L2 regularization.**
    - **Sparse Aware** implementation with automatic handling of missing data values.
    - **Block Structure** to support the parallelization of tree construction.
    - **Continued Training** so that you can further boost an already fitted model on new data.
    
### 3. "Census Income" Dataset to test Light GBM and XGBoost model
![Model Performence]()   
![light GBM ROC Curve]()
![XGBoost ROC Curve]()








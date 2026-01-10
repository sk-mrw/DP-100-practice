# Day1 summary
**Azure ML Workspace Architecture:**
- **Resource Group**: Container for all Azure ML resources
- **Workspace**: Central hub for ML operations
- **Compute Resources**: Where training and inference happen
- **Assets**: Datasets, compute, models, environments, endpoints

**Workspace Navigation (Studio Interface):**
- **Assets Section**: Data, compute, models, environments, components
- **Authoring Section**: Notebooks, AutoML, Designer
- **Manage Section**: Compute, datastores, endpoints

**Key Concepts:**
- Workspace provides collaboration and organization
- Role-based access control (RBAC) for security (Default Owner, Contributor, Reader, AzureML Data Scientist and AzureML Compute Operator)
- Centralized logging and monitoring
- Integration with Azure DevOps and GitHub



**EXAM FOCUS: Task Types**
1. **Classification**: Predict categories
   - Binary: Two classes (yes/no, true/false)
   - Multiclass: Multiple classes (product categories)
   
2. **Regression**: Predict continuous values
   - Examples: Price, temperature, demand
   
3. **Forecasting**: Time-series predictions
   - Requires datetime column
   - Forecasts future values

**Primary Metrics (MEMORIZE THESE):**

**Classification:**
- `accuracy`: Overall correctness
- `AUC_weighted`: Area under ROC curve (weighted by class)
- `precision_score_weighted`: Precision across classes
- `recall_score_weighted`: Recall across classes
- `f1_score_weighted`: Harmonic mean of precision and recall

**Regression:**
- `r2_score`: Coefficient of determination
- `normalized_root_mean_squared_error`: Normalized RMSE
- `normalized_mean_absolute_error`: Normalized MAE
- `spearman_correlation`: Rank correlation

**Forecasting:**
- `normalized_root_mean_squared_error`
- `normalized_mean_absolute_error`
- `r2_score`

**Featurization Options (EXAM CRITICAL):**
- **automatic**: AutoML handles feature engineering
  - Scaling, normalization, encoding
  - Missing value imputation
  - Feature generation
  
- **custom**: Specify transformations
  - Control which features to transform
  - Custom encoding strategies
  
- **off**: No featurization
  - Use when data is pre-processed
  - You handle all transformations

**Validation Types:**
- **k-fold cross-validation**: Split into k folds
  - Default k=5
  - Each fold used once for validation
  
- **Monte Carlo**: Random train/validation splits
  - Multiple iterations with different splits
  
- **train-validation split**: Single split
  - Faster but less robust

**Blocked Algorithms and Allowed Models:**
- Can block specific algorithms from consideration
- Useful when certain models don't fit use case
- Example: Block neural networks for faster training

**Exit Criteria (COST OPTIMIZATION - EXAM FAVORITE):**
- **Metric threshold**: Stop when metric achieved
- **Timeout**: Maximum training time (minutes)
- **Iteration timeout**: Max time per model
- **Max iterations**: Maximum models to try
- **Max concurrent iterations**: Parallel experiments


1. How to create dataset from web/upload


2. AutoML configuration parameters


3. Difference between primary metrics


4. How feature importance helps interpretation


5. Exit criteria impact on training time/cost
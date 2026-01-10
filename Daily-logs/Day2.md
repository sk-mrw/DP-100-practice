**Designer fundamentals**

**EXAM FOCUS: Component Types**

**1. Data Transformation Components:**
- **Select Columns in Dataset**: Choose features for training
- **Clean Missing Data**: Handle null values
  - Remove rows with missing values
  - Replace with mean, median, mode
  - Custom value replacement
  
- **Normalize Data**: Scale features
  - MinMax: Scale to [0,1]
  - ZScore(StandardScalar): Mean=0, StdDev=1
  - Logistic: Sigmoid transformation = 1/(1-e^x)
  
- **Split Data**: Train/test split
  - Stratified sampling (maintains class distribution)
  - Random sampling
  - Recommended split: 70/30 or 80/20

**2. Training Components:**
- **Train Model**: Trains algorithm with data
  - Requires untrained model input
  - Requires training dataset
  - Outputs trained model
  
- **Tune Model Hyperparameters**: Grid search for optimal parameters
  - Defines parameter ranges
  - Uses cross-validation
  - Outputs best model

**3. Scoring Components:**
- **Score Model**: Makes predictions
  - Inputs: Trained model + test data
  - Outputs: Predictions + original data
  
- **Evaluate Model**: Calculates performance metrics
  - Classification: Accuracy, AUC, Precision, Recall
  - Regression: MAE, RMSE, R²

**Pipeline Types (CRITICAL EXAM DIFFERENCE):**

**Training Pipeline:**
- Contains all components
- Includes Train Model, Evaluate Model
- Used during development
- Generates models

**Inference Pipeline (Real-time):**
- Automatically created from training pipeline
- Removes training-specific components
- Adds Web Service Input/Output
- Only includes Score Model
- Used for deployment

**Inference Pipeline (Batch):**
- Similar to real-time but for batch scoring
- Processes multiple records
- Outputs to datastore

**EXAM TIP:** Know what gets removed during inference pipeline creation:
- Evaluation components
- Training data connections
- Cross-validation components

**Published Pipeline:**
- Reusable pipeline with REST endpoint
- Can be triggered programmatically
- Supports parameters

**Pipeline Endpoint:**
- Multiple versions of same pipeline
- Allows A/B testing
- Versioned deployments




Different types of YAML files are used for different purpose

## YAML files for the most common types
### If  YAML is for...:Use this $schema URL
- Data Asset -> https://azuremlschemas.azureedge.net/latest/data.schema.json
- Command Job (Training/Prep) -> https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
- Environment -> https://azuremlschemas.azureedge.net/latest/environment.schema.json
- Pipeline Job -> https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
- Model -> https://azuremlschemas.azureedge.net/latest/model.schema.json
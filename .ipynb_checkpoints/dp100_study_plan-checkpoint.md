# DP-100: 14-Day Intensive Study Plan
## Azure Data Scientist Associate Certification
### Complete Study Guide with Cloud Slices Strategy

---

## Table of Contents

1. [Introduction & Overview](#introduction)
2. [Cloud Slices Best Practices](#cloud-slices)
3. [Phase 1: Automated ML & Designer (Days 1-3)](#phase1)
4. [Phase 2: Notebooks & SDK v2 (Days 4-7)](#phase2)
5. [Phase 3: Deployment & MLOps (Days 8-11)](#phase3)
6. [Phase 4: Mock Exams & Review (Days 12-14)](#phase4)
7. [Critical Exam Tips](#exam-tips)
8. [Quick Reference Guide](#quick-reference)

---

## Introduction & Overview {#introduction}

**Your Profile:**
- Background: Intermediate Python, Beginner Azure
- Study Time: 2 hours every morning
- Timeline: 14 days intensive preparation
- Resources: Azure Cloud Slices (1.5 hour sessions)

**Study Plan Structure:**
This plan is designed by a Microsoft Certified Trainer to maximize your limited study time. Each day combines:
- Hour 1: Theory and preparation (no Cloud Slice)
- Hour 2: Hands-on practice with Cloud Slices

**Exam Overview:**
- Exam Code: DP-100
- Duration: 100 minutes
- Questions: 40-60 scenario-based questions
- Passing Score: 700/1000 (70%)
- Format: Multiple choice, case studies, drag-and-drop

---

## Cloud Slices Best Practices {#cloud-slices}

### Understanding Cloud Slices

**What are Cloud Slices?**
- Pre-configured Azure environments
- Auto-terminate after 90 minutes
- Cannot pause or save state
- Perfect for focused, hands-on labs
- Cost-free practice environment

### Optimal Usage Strategy

**Before Starting Cloud Slice:**
1. Complete all theory reading
2. Have lab instructions ready
3. Create code templates on local machine
4. Set timer for 85 minutes (5-minute buffer)

**During Cloud Slice Session:**
1. Take screenshots of important configurations
2. Copy code snippets to local files
3. Focus on one complete workflow per session
4. Document errors and solutions

**After Cloud Slice Session:**
1. Save all screenshots with descriptive names
2. Organize code snippets by topic
3. Write 5 key learnings in notes
4. Review what worked and what didn't

**Pro Tips:**
- Screenshots disappear after session ends
- Use OneNote or folder structure for organization
- Practice typing code patterns from memory
- Don't try to do too much in one session

---

## PHASE 1: Automated ML & Designer (Days 1-3) {#phase1}

---

### Day 1: Azure ML Workspace Setup & AutoML Classification

#### Hour 1 - Theory & Prep (No Cloud Slice)

**0:00-0:30: Study workspace components**

**Azure ML Workspace Architecture:**
- **Resource Group**: Container for all Azure ML resources
- **Workspace**: Central hub for ML operations
- **Compute Resources**: Where training and inference happen
- **Assets**: Datasets, models, environments, endpoints

**Workspace Navigation (Studio Interface):**
- **Assets Section**: Data, models, environments, components
- **Authoring Section**: Notebooks, AutoML, Designer
- **Manage Section**: Compute, datastores, endpoints

**Key Concepts:**
- Workspace provides collaboration and organization
- Role-based access control (RBAC) for security
- Centralized logging and monitoring
- Integration with Azure DevOps and GitHub

**0:30-1:00: AutoML theory deep-dive**

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

#### Hour 2 - Cloud Slice Hands-On

**LAUNCH CLOUD SLICE - SET TIMER FOR 85 MINUTES**

**0:00-0:10: Quick environment orientation**

Navigate to Azure ML Studio:
1. Go to portal.azure.com
2. Search for "Machine Learning"
3. Select your workspace
4. Click "Launch studio"

Verify:
- Workspace is accessible
- Compute resources available
- Data section is visible

**0:10-0:40: AutoML Classification Experiment**

**Step 1: Create Dataset**

```
Navigation: Data → Create → From web files

Configuration:
- Name: titanic-dataset
- Type: Tabular
- Web URL: Use sample dataset or upload CSV
```

**Sample Dataset Structure:**
- Features: PassengerClass, Age, Fare, Sex, etc.
- Target: Survived (0 or 1)

**Screenshot Checklist:**
- Data preview showing rows and columns
- Schema with data types
- Profile statistics (if available)

**Step 2: Configure AutoML Job**

```
Navigation: AutoML → New AutoML job → Classification

Task Settings:
- Task type: Classification
- Dataset: titanic-dataset
- Target column: Survived
- Primary metric: AUC_weighted

Additional Configuration:
- Explain best model: Yes ✓ (CRITICAL for exam)
- Validation type: k-fold cross-validation
- Number of folds: 5

Exit Criteria:
- Training job time (hours): 0.25 (15 minutes)
- Metric score threshold: 0.85
- Max concurrent iterations: 2

Featurization:
- Automatic ✓
```

**Screenshot Checklist:**
- Task configuration page
- Featurization settings
- Exit criteria settings
- Validation configuration

**Submit the job!**

**0:40-1:15: While AutoML Runs**

**Explore Studio Navigation:**

1. **Compute Section:**
   - View available compute instances
   - Check compute clusters
   - Understand VM sizes and pricing

2. **Data Assets:**
   - See registered datasets
   - Understand versioning
   - Check data profiles

3. **Notebooks:**
   - Explore sample notebooks
   - Understand Jupyter integration

4. **Components:**
   - Pre-built pipeline components
   - Custom component creation

**Monitor AutoML Job:**
- Navigate to Jobs section
- Click on your AutoML job
- Watch iterations complete
- Check Models tab for completed models
- View metrics as they're logged

**Screenshot Checklist:**
- Compute configuration page showing VM sizes
- Job monitoring page showing iterations
- Running experiments list

**1:15-1:30: Review Results**

**When AutoML Completes:**

1. **Best Model Summary:**
   - Which algorithm was selected?
   - What metric did it achieve?
   - Training time

2. **Feature Importance:**
   - Navigate to "Explanations" tab
   - View global feature importance chart
   - Identify top contributing features

3. **Model Details:**
   - Algorithm parameters
   - Validation scores
   - Confusion matrix (if available)

**Screenshot Checklist:**
- Best model summary page
- Feature importance chart
- Performance metrics
- Algorithm details

**Key Exam Concept:**
Understand WHY that model was selected:
- Best metric score
- Within time constraints
- Validation performance

**Post-Session (10 minutes):**

Document in your notes:
1. Best model algorithm: _______
2. AUC score achieved: _______
3. Top 3 important features: _______
4. Total training time: _______
5. Number of models tried: _______

**Key Learnings to Write:**
1. How to create dataset from web/upload
2. AutoML configuration parameters
3. Difference between primary metrics
4. How feature importance helps interpretation
5. Exit criteria impact on training time/cost

---

### Day 2: Azure ML Designer - Complete ML Pipeline

#### Hour 1 - Theory & Prep

**0:00-0:40: Designer fundamentals**

**EXAM FOCUS: Component Types**

**1. Data Transformation Components:**
- **Select Columns in Dataset**: Choose features for training
- **Clean Missing Data**: Handle null values
  - Remove rows with missing values
  - Replace with mean, median, mode
  - Custom value replacement
  
- **Normalize Data**: Scale features
  - MinMax: Scale to [0,1]
  - ZScore: Mean=0, StdDev=1
  - Logistic: Sigmoid transformation
  
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

**0:40-1:00: Pre-plan your pipeline**

**On Paper, Draw This Flow:**

```
[Automobile Dataset]
        ↓
[Select Columns] (remove ID, unnecessary features)
        ↓
[Clean Missing Data] (replace with mean)
        ↓
[Normalize Data] (MinMax normalization)
        ↓
[Split Data] (70% train, 30% test)
        ↓        ↓
    [Train]  [Score]
     Model    Model
        ↓        ↓
    [Score] → [Evaluate]
     Model     Model
```

**Choose Your Algorithm:**
- **Two-Class Logistic Regression** (for binary classification)
- **Linear Regression** (for regression tasks)

**Plan Your Configurations:**
- Which columns to select?
- How to handle missing data?
- Normalization method?
- Train/test split ratio?

#### Hour 2 - Cloud Slice Hands-On

**LAUNCH CLOUD SLICE**

**0:00-0:50: Build Training Pipeline**

**Step 1: Create New Pipeline**

```
Navigation: Designer → + New pipeline → Classic prebuilt

Pipeline Name: automobile-price-prediction
Compute: Select available compute (or create small cluster)
```

**Step 2: Drag Components (Follow This Order)**

**a) Add Dataset:**
- Search "Automobile price data (Raw)"
- Drag to canvas

**b) Select Columns in Dataset:**
- Connect from dataset
- Configure: Remove "normalized-losses" column
- Keep all others

**c) Clean Missing Data:**
- Connect from Select Columns
- Configure:
  - Columns to clean: All columns
  - Cleaning mode: Replace with mean
  
**d) Normalize Data:**
- Connect from Clean Missing Data
- Configure:
  - Transformation method: MinMax
  - Columns to transform: All numeric columns

**e) Split Data:**
- Connect from Normalize Data
- Configure:
  - Splitting mode: Split Rows
  - Fraction: 0.7 (70% training)
  - Randomized split: Yes ✓
  - Stratified split: No

**f) Linear Regression:**
- Drag "Linear Regression" algorithm
- Leave default parameters
- DO NOT connect yet (just place on canvas)

**g) Train Model:**
- Connect: Linear Regression → Train Model (left input)
- Connect: Split Data (left output) → Train Model (right input)
- Configure: Label column = "price"

**h) Score Model:**
- Connect: Train Model → Score Model (left input)
- Connect: Split Data (right output) → Score Model (right input)

**i) Evaluate Model:**
- Connect: Score Model → Evaluate Model

**Critical Connection Pattern (EXAM TESTED):**
```
Split Data has TWO outputs:
- Left (Results dataset1): 70% → Train Model
- Right (Results dataset2): 30% → Score Model

Train Model has TWO inputs:
- Left: Untrained model (Linear Regression)
- Right: Training data (from Split)

Score Model has TWO inputs:
- Left: Trained model (from Train Model)
- Right: Test data (from Split Data right output)
```

**Screenshot Checklist:**
- Complete pipeline layout
- All connections properly made
- Component configurations

**Step 3: Configure Components**

For each component, verify configuration:
- Train Model: Label column selected
- Split Data: 0.7 fraction
- Clean Missing Data: Replace with mean

**0:50-1:10: Submit Pipeline Job**

**Submit:**
```
Experiment name: designer-experiment
Run description: Automobile price prediction training
```

**Monitor Execution:**
- Watch components turn green as they complete
- Check for any errors (red)
- Estimated time: 5-10 minutes

**Screenshot Checklist:**
- Running pipeline (components turning green)
- Completed pipeline (all green checkmarks)

**1:10-1:25: Create Inference Pipeline**

**CRITICAL EXAM TOPIC:**

**After training completes:**
```
Click: "Create inference pipeline" → "Real-time inference pipeline"
```

**Observe Automatic Changes:**

**Removed:**
- Evaluate Model component
- Split Data component (no longer needed)
- Training data connections

**Added:**
- Web Service Input
- Web Service Output

**Retained:**
- Score Model (for predictions)
- Data transformation components (Clean, Normalize)

**Modified:**
- Training dataset replaced with web service input
- Only scoring path remains

**Screenshot Checklist:**
- Inference pipeline structure
- Web Service Input/Output placement
- Simplified component flow

**1:25-1:30: Review Evaluation Metrics**

**In Training Pipeline:**
```
Right-click "Evaluate Model" → Visualize
```

**Check Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Relative Squared Error
- R² Score
- Coefficient of Determination

**Screenshot Checklist:**
- Evaluation metrics visualization
- Performance chart

**Post-Session:**

**Document These Differences:**

| Aspect | Training Pipeline | Inference Pipeline |
|--------|------------------|-------------------|
| Purpose | Model development | Production scoring |
| Contains | Train, Evaluate | Score only |
| Inputs | Dataset | Web Service Input |
| Outputs | Metrics, Model | Predictions |
| Components | All transformation + training | Transformation + scoring |

**Key Learnings:**
1. Component connection logic (especially Split Data)
2. Training vs Inference pipeline purposes
3. Which components get removed in inference
4. How to configure each component type
5. Reading evaluation metrics

---

### Day 3: Designer Deployment & AutoML Forecasting

#### Hour 1 - Theory & Prep

**0:00-0:35: Deployment concepts**

**EXAM FOCUS: Real-time Endpoints**

**Managed Online Endpoints:**
- **Endpoint**: Stable HTTPS URL for predictions
- **Deployment**: Actual model serving infrastructure
- **Relationship**: One endpoint → Multiple deployments

**Endpoint Properties:**
- Unique name (within workspace)
- Authentication: Key-based or Token-based
- Swagger documentation (auto-generated)
- Monitoring and logging

**Deployment Properties:**
- Instance type: VM size (Standard_DS2_v2, etc.)
- Instance count: Number of VMs
- Environment: Docker container with dependencies
- Code configuration: Scoring script

**Blue-Green Deployment Strategy:**
```
Endpoint: my-endpoint
├── Deployment "blue" (current production) - 100% traffic
└── Deployment "green" (new version) - 0% traffic

After testing:
├── Deployment "blue" - 50% traffic
└── Deployment "green" - 50% traffic

Finally:
├── Deployment "blue" - 0% traffic (can delete)
└── Deployment "green" - 100% traffic
```

**Traffic Allocation Syntax:**
```python
endpoint.traffic = {"blue": 90, "green": 10}
endpoint.traffic = {"blue": 50, "green": 50}
endpoint.traffic = {"green": 100}
```

**Batch Endpoints:**
- For large-scale scoring jobs
- Not always-on (cost-effective)
- Processes files or datasets
- Outputs to datastore
- Scheduled execution

**Batch vs Real-time Comparison:**

| Feature | Real-time | Batch |
|---------|-----------|-------|
| Latency | Milliseconds | Minutes/Hours |
| Input | Single/few records | Large datasets |
| Cost | Always-on | Pay per execution |
| Use Case | Interactive apps | Scheduled reports |
| Infrastructure | Always running | On-demand |

**Pipeline Endpoints:**
- Reusable pipeline with REST endpoint
- Can pass parameters at runtime
- Versioning support
- Programmatic triggering

**0:35-1:00: AutoML Forecasting specifics**

**CRITICAL EXAM CONCEPTS:**

**Time Column Selection:**
- Must be datetime type
- Required for all forecasting tasks
- Examples: "date", "timestamp", "datetime_col"

**Time Series ID Columns:**
- For multiple time series in one dataset
- Example: Store ID, Product ID, Region
- Each unique ID gets separate forecast
- Optional but common in practice

**Forecast Horizon:**
- How many periods ahead to predict
- Example: 12 (predict next 12 months)
- Must match business need
- Affects model complexity

**Target Lags:**
- Use past values as features
- Lag 1: Previous period
- Lag 7: Same day last week
- Auto-detect or manual specification

**Rolling Windows:**
- Aggregations over windows
- Example: 7-day rolling average
- Helps capture trends
- Smooths noise

**Seasonality:**
- Auto-detect patterns (recommended)
- Manual: Specify season length
- Examples: 7 (weekly), 12 (monthly), 52 (yearly)

**Frequency:**
- Granularity of time series
- Options: Hourly, Daily, Weekly, Monthly, Yearly
- Must match data structure
- Determines forecast intervals

**Forecasting-Specific Configuration:**
```
Task: Forecasting
Time column: datetime
Time series ID columns: [store_id, product_id]
Forecast horizon: 12
Frequency: Daily
Target lags: Auto
Rolling window size: Auto
Seasonality: Auto
```

#### Hour 2 - Cloud Slice Hands-On

**LAUNCH CLOUD SLICE**

**0:00-0:25: Deploy Designer Pipeline**

**Option A: If Yesterday's Pipeline Available**

Navigate to: Designer → Your inference pipeline from Day 2

**Option B: Create Simple Pipeline Quickly**

```
[Sample Dataset] → [Score Model] → [Web Service Output]
(Use any pre-trained model component if available)
```

**Deploy to Real-time Endpoint:**

```
Click: "Deploy" button

Configuration:
- Compute type: Azure Container Instance (ACI)
- Endpoint name: designer-endpoint-test
- Description: Test deployment from designer
- Authentication type: Key
- Enable Application Insights: Yes ✓
```

**Deployment Process:**
- Initializing: Creating container
- Building: Installing dependencies
- Deploying: Starting service
- Healthy: Ready to use

Time: 5-10 minutes

**While Deploying, Navigate to Endpoints:**
```
Navigate: Endpoints → Real-time endpoints

You'll see: designer-endpoint-test (Updating)
```

**Screenshot Checklist:**
- Deployment configuration screen
- Endpoint status (Updating → Healthy)

**0:25-1:00: AutoML Forecasting Experiment**

**Step 1: Prepare Time-Series Dataset**

**Option A: Use Sample Energy Demand Data**
```
Search for: "Energy demand time series" sample

Should have columns like:
- timestamp (datetime)
- demand (numeric)
```

**Option B: Create Simple Dataset**
```python
# If using Notebook temporarily:
import pandas as pd
import numpy as np

dates = pd.date_range('2023-01-01', periods=365, freq='D')
df = pd.DataFrame({
    'date': dates,
    'demand': 100 + 20*np.sin(np.arange(365)*2*np.pi/7) + np.random.randn(365)*5
})
df.to_csv('energy_demand.csv', index=False)
```

Register as dataset.

**Step 2: Configure AutoML Forecasting**

```
Navigate: AutoML → New AutoML job → Forecasting

Task Configuration:
- Dataset: energy-demand-dataset
- Target column: demand
- Time column: date (MUST select!)

Forecasting Settings:
- Forecast horizon: 12
- Time series ID columns: (leave empty if single series)
- Frequency: Daily
- Target lags: Auto
- Rolling window: Auto
- Seasonality: Auto

Primary Metric: normalized_root_mean_squared_error

Validation:
- Time series cross-validation ✓
- Number of CV folds: 3

Exit Criteria:
- Training job time: 10 minutes
- Metric threshold: 0.15 (lower is better for RMSE)
- Max concurrent iterations: 2

Featurization: Automatic
```

**Screenshot Checklist:**
- Time column selection (CRITICAL)
- Forecast horizon setting
- Frequency configuration
- Forecasting-specific parameters

**Submit the Job!**

**Monitor:**
- Job status in Jobs section
- Iterations completing
- Forecasting models being tried

**1:00-1:20: Test Deployed Endpoint**

**Navigate to Deployed Endpoint:**
```
Endpoints → Real-time endpoints → designer-endpoint-test
```

**Go to "Test" Tab:**

**Prepare Input JSON:**
```json
{
  "data": [
    [value1, value2, value3, ...]
  ]
}
```

**Example for Automobile Price:**
```json
{
  "data": [
    [3, 100, 8, 150, 4000, 15, 90, 2.5]
  ]
}
```

**Click "Test"**

**Expected Response:**
```json
{
  "Results": [
    15234.56
  ]
}
```

**Screenshot Checklist:**
- Test input JSON
- Successful prediction output
- Response time

**Get Endpoint Details:**
```
Go to "Consume" tab:

Scoring URI: https://...azureml.net/score
Authentication: Key
Primary Key: *********************
```

**Screenshot Checklist:**
- Scoring URI
- Authentication method
- Sample code provided

**1:20-1:30: Review AutoML Forecasting Results**

**Check Job Status:**
```
Navigate: Jobs → Your forecasting job
```

**If Complete, Review:**

1. **Best Model:**
   - Algorithm selected (often ARIMA, Prophet, or ensembles)
   - nRMSE achieved
   - Training time

2. **Forecast Visualization:**
   - Navigate to "Explanations" or "Metrics"
   - Look for forecast vs actual plot
   - Check residuals

3. **Metrics:**
   - Normalized RMSE
   - Normalized MAE
   - R² score
   - MAPE (Mean Absolute Percentage Error)

**Screenshot Checklist:**
- Best forecasting model
- Forecast vs actual visualization
- Performance metrics specific to forecasting

**Post-Session:**

**Key Differences to Document:**

**Forecasting vs Classification/Regression:**

| Aspect | Forecasting | Classification | Regression |
|--------|------------|----------------|-----------|
| Target | Time-series values | Categories | Continuous values |
| Required Column | Datetime | None | None |
| Special Config | Horizon, frequency | None | None |
| Validation | Time-series CV | K-fold CV | K-fold CV |
| Metrics | nRMSE, nMAE | AUC, Accuracy | RMSE, R² |

**Endpoint Authentication Methods:**
1. **Key-based**: Static keys (primary/secondary)
2. **Token-based**: Azure AD authentication
3. **No authentication**: Not recommended for production

**Key Learnings:**
1. Forecasting requires datetime column selection
2. Forecast horizon determines prediction length
3. Endpoint deployment takes 5-10 minutes
4. Test tab allows quick validation
5. Scoring URI is the production endpoint

---

## PHASE 2: Notebooks & SDK v2 (Days 4-7) {#phase2}

---

### Day 4: SDK v2 Workspace Connection & Data Assets

#### Hour 1 - Theory & Prep

**0:00-0:40: SDK v2 Architecture**

**CRITICAL EXAM FOCUS: MLClient is Central**

**SDK v1 vs SDK v2 (Know the Difference):**

| Aspect | SDK v1 | SDK v2 |
|--------|--------|--------|
| Main Class | Workspace | MLClient |
| Style | Object-oriented | Resource-based |
| Config | workspace.from_config() | MLClient.from_config() |
| Jobs | Run | Job |
| Current Status | Deprecated | Current |

**MLClient Authentication Methods:**

**1. DefaultAzureCredential (Recommended):**
```python
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential,
    subscription_id="<sub-id>",
    resource_group_name="<rg-name>",
    workspace_name="<ws-name>"
)
```

**2. InteractiveBrowserCredential:**
```python
from azure.identity import InteractiveBrowserCredential

credential = InteractiveBrowserCredential()
ml_client = MLClient(credential, ...)
```

**3. From Config File:**
```python
ml_client = MLClient.from_config(DefaultAzureCredential())
```

**Configuration Parameters:**
- `subscription_id`: Azure subscription GUID
- `resource_group_name`: Resource group containing workspace
- `workspace_name`: Azure ML workspace name

**Data Assets Types (EXAM CRITICAL):**

**1. uri_file:**
- Single file (CSV, JSON, image, etc.)
- Direct file path
- Example: training.csv

**2. uri_folder:**
- Directory of files
- Multiple files in same location
- Example: images folder with 1000 images

**3. mltable:**
- Structured tabular data
- Schema definition
- Optimized for large datasets
- MLTable YAML specification

**When to Use Each:**

| Type | Use Case | Example |
|------|----------|---------|
| uri_file | Single dataset file | train.csv, model.pkl |
| uri_folder | Multiple related files | image dataset, text corpus |
| mltable | Large tabular data | Big CSV with schema |

**Datastores (EXAM TESTED):**

**Default Datastore:**
- Every workspace has `workspaceblobstore`
- Azure Blob Storage
- Automatically created
- Used if no datastore specified

**Supported Datastore Types:**
- **Azure Blob Storage**: Most common
- **Azure Data Lake Gen2**: Big data scenarios
- **Azure Files**: File shares
- **Azure SQL**: Relational data

**Datastore Path Format:**
```
azureml://datastores/<datastore-name>/paths/<path-to-data>
```

**Example:**
```
azureml://datastores/workspaceblobstore/paths/data/train.csv
```

**0:40-1:00: Prepare Code Snippets**

**Write these patterns on paper or document:**

**Pattern 1: MLClient Connection**
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<sub-id>",
    resource_group_name="<rg-name>",
    workspace_name="<ws-name>"
)

# Verify connection
print(ml_client.workspace_name)
```

**Pattern 2: Data Asset Registration (uri_file)**
```python
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_data = Data(
    path="path/to/file.csv",
    type=AssetTypes.URI_FILE,
    description="Training dataset",
    name="my-training-data",
    version="1"
)

ml_client.data.create_or_update(my_data)
```

**Pattern 3: Data Asset Registration (uri_folder)**
```python
folder_data = Data(
    path="path/to/folder",
    type=AssetTypes.URI_FOLDER,
    description="Image dataset",
    name="image-data",
    version="1"
)

ml_client.data.create_or_update(folder_data)
```

**Pattern 4: Data Asset from Datastore**
```python
datastore_data = Data(
    path="azureml://datastores/workspaceblobstore/paths/data/file.csv",
    type=AssetTypes.URI_FILE,
    name="datastore-data",
    version="1"
)

ml_client.data.create_or_update(datastore_data)
```

**Pattern 5: Retrieve Data Asset**
```python
# Get specific version
data_asset = ml_client.data.get(name="my-data", version="1")

# Get latest version
data_asset = ml_client.data.get(name="my-data", label="latest")

print(f"Name: {data_asset.name}")
print(f"Path: {data_asset.path}")
print(f"Version: {data_asset.version}")
```

####
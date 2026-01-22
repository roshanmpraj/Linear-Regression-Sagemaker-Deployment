
---

# 🔵 COMPLETE END-TO-END EXECUTION MANUAL

## Airbnb Price Prediction using AWS SageMaker

*(Zero assumptions, click-by-click, copy-paste ready)*

---

# SECTION 0 — WHAT YOU NEED BEFORE STARTING

You MUST have:

1. An **AWS account** (Free Tier is fine)
2. A **GitHub account**
3. A **GitHub repository created** (empty repo is fine)
4. Dataset file: `airbnb.csv` on your local machine

Nothing else is required.

---

# SECTION 1 — LOGIN & NAVIGATION (NO GUESSING)

## Step 1.1 — Login to AWS

1. Open browser
2. Go to: **[https://console.aws.amazon.com](https://console.aws.amazon.com)**
3. Login using your AWS credentials

You should see the **AWS Management Console**.

---

## Step 1.2 — Go to SageMaker

1. In the AWS Console **top search bar**
2. Type: `SageMaker`
3. Click **Amazon SageMaker**

You are now on the **SageMaker Dashboard**.

---

## Step 1.3 — Open SageMaker Studio

1. In the **left navigation panel**
2. Click **Studio**
3. Click **Open Studio**

⏳ First time may take 1–2 minutes.

When loaded, you are on **SageMaker Studio Home**.

---

# SECTION 2 — OPEN JUPYTERLAB (WHERE ALL WORK HAPPENS)

## Step 2.1 — Open JupyterLab

On the SageMaker Studio Home page:

1. Locate **JupyterLab**
2. Click **View JupyterLab spaces**
3. Click **Create space** (if none exists)

### Choose:

* Instance type: `ml.t3.medium`
* Image: Default

4. Click **Create**
5. After creation, click **Open**

You are now inside **JupyterLab**.

👉 **THIS is your working environment**

---

# SECTION 3 — UNDERSTAND THE INTERFACE (IMPORTANT)

Inside JupyterLab you will see:

### Left Panel

* File browser (folders & files)

### Main Area

* Editors
* Notebooks
* Terminals

You will use **TWO things only**:

1. **Terminal**
2. **Notebook**

---

# SECTION 4 — OPEN TERMINAL (FOR GIT & FILES)

## Step 4.1 — Open Terminal

1. Click **+ (Launcher)** at top
2. Scroll to **Other**
3. Click **Terminal**

A black terminal window opens.

📌 **This terminal is a Linux machine managed by AWS.**

---

# SECTION 5 — GITHUB AUTHENTICATION (EXACT STEPS)

## Step 5.1 — Create GitHub Token

On **github.com**:

1. Click profile picture → **Settings**
2. Click **Developer settings**
3. Click **Personal access tokens**
4. Click **Tokens (classic)**
5. Click **Generate new token**
6. Select scope:

   * ✅ `repo`
7. Click **Generate**
8. COPY THE TOKEN (you will not see it again)

---

## Step 5.2 — Clone Repo in SageMaker

📍 **Run inside Terminal**

```bash
cd ~
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>
```

When prompted:

* Username → GitHub username
* Password → **PASTE TOKEN**

✅ Repo is now on SageMaker.

---

# SECTION 6 — CREATE PROJECT STRUCTURE

📍 **Terminal**

```bash
mkdir -p src data
touch src/train.py
touch src/inference.py
touch requirements.txt
```

Your folder MUST look like this:

```
<repo-name>/
├── src/
│   ├── train.py
│   ├── inference.py
├── data/
├── requirements.txt
```

---

# SECTION 7 — UPLOAD DATASET (CLICK STEPS)

## Step 7.1 — Upload CSV

1. In **left file browser**
2. Click `data/`
3. Click **Upload**
4. Select `airbnb.csv` from your laptop
5. Confirm upload

Final path MUST be:

```
data/airbnb.csv
```

---

# SECTION 8 — ADD PYTHON CODE (WHERE TO WRITE)

## Step 8.1 — Open train.py

1. Double-click `src/train.py`
2. Paste the following **FULL CODE**
3. Save (`Ctrl + S`)

```python
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    data_dir = "/opt/ml/input/data/train"
    model_dir = "/opt/ml/model"

    df = pd.read_csv(os.path.join(data_dir, "airbnb.csv"))

    features = [
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "review_scores_rating"
    ]

    X = df[features].fillna(df[features].median())
    y = df["log_price"].fillna(df["log_price"].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("RMSE:", mean_squared_error(y_test, preds, squared=False))
    print("R2:", r2_score(y_test, preds))

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    main()
```

---

## Step 8.2 — Open inference.py

1. Double-click `src/inference.py`
2. Paste:

```python
import json
import joblib
import numpy as np
import os

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, content_type):
    if content_type == "application/json":
        return np.array(json.loads(request_body))
    raise ValueError("Unsupported content type")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    return json.dumps(prediction.tolist())
```

Save file.

---

## Step 8.3 — Open requirements.txt

Paste:

```text
pandas
numpy
scikit-learn
joblib
```

Save.

---

# SECTION 9 — COMMIT CODE TO GITHUB

📍 **Terminal**

```bash
git add .
git commit -m "Initial SageMaker training and inference code"
git push origin main
```

---

# SECTION 10 — CREATE NOTEBOOK (WHERE TO RUN CODE)

## Step 10.1 — Create Notebook

1. File → New → Notebook
2. Kernel: **Python 3 (Data Science)**
3. Name it:

```
train_and_deploy.ipynb
```

---

# SECTION 11 — UPLOAD DATA TO S3 (NOTEBOOK CELL)

```python
import sagemaker

session = sagemaker.Session()
bucket = session.default_bucket()

session.upload_data(
    path="data/airbnb.csv",
    bucket=bucket,
    key_prefix="airbnb/data/train"
)

print(bucket)
```

---

# SECTION 12 — RUN TRAINING JOB (NOTEBOOK CELL)

```python
from sagemaker.sklearn.estimator import SKLearn
from sagemaker import get_execution_role

role = get_execution_role()

estimator = SKLearn(
    entry_point="train.py",
    source_dir="src",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    output_path=f"s3://{bucket}/airbnb/models"
)

estimator.fit({"train": f"s3://{bucket}/airbnb/data/train"})
```

⏳ Wait until status = **Completed**

---

# SECTION 13 — DEPLOY ENDPOINT

```python
from sagemaker.sklearn.model import SKLearnModel

model = SKLearnModel(
    model_data=estimator.model_data,
    role=role,
    entry_point="inference.py",
    source_dir="src",
    framework_version="1.2-1",
    py_version="py3"
)

predictor = model.deploy(
    instance_type="ml.t2.medium",
    initial_instance_count=1,
    endpoint_name="airbnb-linear-regression-prod"
)
```

Wait until endpoint = **InService**

---

# SECTION 14 — TEST PREDICTION

```python
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

predictor.serializer = JSONSerializer()
predictor.deserializer = JSONDeserializer()

sample = [[2, 1.0, 1, 1, 95.0]]
prediction = predictor.predict(sample)

print("Prediction:", prediction)
```

---

# SECTION 15 — LOGS (WHERE TO CHECK)

AWS Console → CloudWatch:

* Training:

  ```
  /aws/sagemaker/TrainingJobs
  ```
* Endpoint:

  ```
  /aws/sagemaker/Endpoints/airbnb-linear-regression-prod
  ```

---

# SECTION 16 — CLEANUP (MANDATORY)

```python
predictor.delete_endpoint()
```

Also:

* Stop JupyterLab app
* Verify no endpoints running




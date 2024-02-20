# End to End Chest Cancer Classification

## Overview

This project demonstrates a complete workflow for chest X-ray cancer classification utilizing the power of Machine Learning and MLflow for experiment tracking and logging. By leveraging readily available chest X-ray dataset and state-of-the-art deep learning models to build a robust and accurate classification system. MLflow seamlessly integrates into the process, capturing experiment details, model metrics, and artifacts, enabling reproducibility and insightful analysis.

## Features

- **Pipeline Tracking :** DVC implemented to track artifacts

- **FastAPI :** Developed API using FastAPI for inference

- **MLflow :**  USed Mlflow to perform experiment tracking.


## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/pks916/endtoendml.git
   ```

2. **Create new environment**
```
conda create -n endtoendml
```
3. **Activate the environment**
```
conda activate endtoendml
```
3. **Libraries installation & Setup**
```
pip install -r requirements.txt
```

## Usage
```
uvicorn routes:app
```

## Pipeline

![Pipeline](https://raw.githubusercontent.com/pks916/endtoendml/main/blob/pipeline.png)

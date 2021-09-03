# SHAP Model Explainability

Initial Stages: 
- Import the cleaned dataframe (after feature engineering and data cleaning)

```
import pickle
import shap
shap.initjs()
import numpy as np
import matplotlib.pyplot as plt
from alibi.explainers import TreeShap
from functools import partial
from itertools import product, zip_longest
from scipy.special import expit
invlogit=expit
from sklearn.metrics import accuracy_score, confusion_matrix
from timeit import default_timer as timer
```

```
import os
import pandas as pd
import numpy as np
import mlflow
import evalml
import shap
import databricks.koalas as ks
from pyspark.sql import SparkSession
from evalml.pipelines.components.transformers import Transformer
import mleap.pyspark
from mleap.pyspark.spark_support import SimpleSparkSerializer
from pyspark.ml import PipelineModel
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import mleap.pyspark
from mleap.pyspark.spark_support import SimpleSparkSerializer
from pyspark.ml import PipelineModel

dbutils.fs.cp("dbfs:/mnt/models/mleap_model-json.zip", "file:/tmp/mleap_python_model_import/mleap_model-json.zip")

RECmodel = PipelineModel.deserializeFromBundle("file:/tmp/mleap_python_model_import/mleap_model-json.zip")

file_loc = "dbfs:/mnt/data/preppedData"
df = spark.read.format("delta").load(file_loc)
display(df)
```

```
#Feature Importance
pipeline.feature_importance
```




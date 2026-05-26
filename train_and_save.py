# ============================================
# Q6 — Train and Save Model to S3
# ============================================

# ---- Section 1: Imports ----
import os
import boto3
import joblib
import urllib3
import numpy as np
import onnx
from botocore.client import Config
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---- Section 2: Data Connection ----
# key_id      =
# secret_key  =
# region      =
# endpoint    =
# bucket_name =

# ---- Section 3: Train Model ----
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Training complete. Accuracy: {accuracy:.2%}")

# ---- saving m1 model ----
# ADD CODE HERE


# --- converting to onnx --- 

loaded_model = joblib.load("abcd.joblib")
initial_type = [('float_input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(loaded_model, initial_types=initial_type)
with open('abcd.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
print("Converted to real ONNX ✅")


# ---- Section 5: Upload to S3 ----
s3 = boto3.client(
    's3',
    region_name=region or 'us-east-1',
    aws_access_key_id=key_id,
    aws_secret_access_key=secret_key,
    endpoint_url=endpoint,
    config=Config(signature_version='s3v4'),
    verify=False
)
s3.upload_file("abcd.onnx", bucket_name, "abcd.onnx")
print("Model uploaded to S3 as abcd.onnx ✅")

# ---- Section 6: Verify ----
response = s3.list_objects_v2(Bucket=bucket_name, Prefix="abcd")
for obj in response.get('Contents', []):
    print(f"Found in S3: {obj['Key']} ({obj['Size']} bytes)")

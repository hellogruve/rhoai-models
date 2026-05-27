import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import boto3, os
from botocore.client import Config
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create a simple linear classifier ONNX using standard ops
# OpenVINO fully supports these (MatMul + Add)
np.random.seed(42)
W = np.random.randn(4, 3).astype(np.float32)
b = np.random.randn(3).astype(np.float32)

W_init = numpy_helper.from_array(W, name='W')
b_init = numpy_helper.from_array(b, name='b')

matmul = helper.make_node('MatMul', ['float_input', 'W'], ['matmul_out'])
add    = helper.make_node('Add', ['matmul_out', 'b'], ['output'])

graph = helper.make_graph(
    [matmul, add],
    'iris_classifier',
    [helper.make_tensor_value_info('float_input', TensorProto.FLOAT, [None, 4])],
    [helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 3])],
    [W_init, b_init]
)

onnx_model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid('', 13)]
)
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, 'efgh.onnx')
print("ONNX created and validated ✅")

# Upload to S3
s3 = boto3.client('s3',
    region_name=os.getenv('AWS_DEFAULT_REGION') or 'us-east-1',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    endpoint_url=os.getenv('AWS_S3_ENDPOINT'),
    config=Config(signature_version='s3v4'),
    verify=False)

s3.upload_file('efgh.onnx', os.getenv('AWS_S3_BUCKET'), 'efgh.onnx')
print("Uploaded as efgh.onnx ✅")

r = s3.list_objects_v2(Bucket=os.getenv('AWS_S3_BUCKET'), Prefix='efgh')
for obj in r.get('Contents', []):
    print(f"S3: {obj['Key']} -- {obj['Size']} bytes")

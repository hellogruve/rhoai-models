import requests
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Replace with your actual inference endpoint from RHOAI Dashboard
# Models tab -> efgh-model -> copy icon next to inference endpoint
INFERENCE_URL = "https://efgh-model-ai267-exam.apps.ocp4.example.com/v2/models/efgh-model/infer"

# Test samples: [sepal_length, sepal_width, petal_length, petal_width]
test_samples = {
    "setosa":     [5.1, 3.5, 1.4, 0.2],
    "versicolor": [6.0, 2.9, 4.5, 1.5],
    "virginica":  [6.3, 3.3, 6.0, 2.5]
}

classes = {0: "setosa", 1: "versicolor", 2: "virginica"}

print("=== Inference Endpoint Test ===\n")

for expected, features in test_samples.items():
    payload = {
        "inputs": [{
            "name": "float_input",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": [features]
        }]
    }

    r = requests.post(INFERENCE_URL, json=payload, verify=False)

    if r.status_code == 200:
        outputs = r.json()["outputs"][0]["data"]
        predicted_class = outputs.index(max(outputs))
        print(f"Input:    {features}")
        print(f"Expected: {expected}")
        print(f"Scores:   {[round(s,3) for s in outputs]}")
        print(f"Status: HTTP {r.status_code} ✅\n")
    else:
        print(f"Input:  {features}")
        print(f"Status: HTTP {r.status_code} ❌")
        print(f"Error:  {r.text}\n")

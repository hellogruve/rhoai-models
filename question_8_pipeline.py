from typing import List, NamedTuple
from kfp import dsl, compiler

PYTHON_IMAGE = "registry.access.redhat.com/ubi9/python-39:latest"

@dsl.component(base_image=PYTHON_IMAGE)
def process_data() -> NamedTuple("outputs", texts=List[str], labels=List[int]):
    dataset = [
        ("I love this!", "positive"), ("I hate this!", "negative"),
        ("This is awesome!", "positive"), ("This is terrible!", "negative"),
        ("Great experience.", "positive"), ("Bad experience.", "negative"),
        ("I like it.", "positive"), ("I don't like it.", "negative"),
    ]
    texts = [s[0].strip() for s in dataset]
    labels = [1 if s[1] == "positive" else 0 for s in dataset]
    outputs = NamedTuple("outputs", texts=List[str], labels=List[int])
    return outputs(texts, labels)

@dsl.component(base_image=PYTHON_IMAGE)
def train_model(texts: list, labels: list) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42)
    pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

@dsl.component(base_image=PYTHON_IMAGE)
def verify_accuracy(accuracy: float, threshold: float):
    import sys
    if accuracy >= threshold:
        print(f"Model trained successfully. Accuracy: {accuracy * 100:.2f}%")
    else:
        print(f"Accuracy {accuracy * 100:.2f}% below threshold {threshold * 100:.2f}%")
        sys.exit(1)

@dsl.pipeline(name="sentiment-analysis")
def pipeline():
    data_task = process_data()
    texts = data_task.outputs["texts"]
    labels = data_task.outputs["labels"]
    train_task = train_model(texts=texts, labels=labels)
    accuracy = train_task.output
    verify_accuracy(accuracy=accuracy, threshold=0.5)

if __name__ == "__main__":
    outfile = "pipeline.yaml"
    compiler.Compiler().compile(pipeline, outfile)
    print(f"Pipeline compiled.\nImport '{outfile}' into RHOAI Dashboard.")

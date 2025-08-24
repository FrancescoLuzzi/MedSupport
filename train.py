from sentence_transformers.evaluation import TripletEvaluator, SimilarityFunction
import polars as pd
from pathlib import Path

# Load triplets from the AllNLI dataset
max_samples = 1000


training_data_folder = Path("./data")
training_dataset_file = training_data_folder / "lista_farmaci_equivalenti.csv"

dataset = pd.read_csv(training_dataset_file, separator=";")
print(dataset["Principio attivo"].unique())

# Initialize the evaluator
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    main_distance_function=SimilarityFunction.COSINE,
    name=f"all-nli-{max_samples}-dev",
)
# Run evaluation manually:
# print(dev_evaluator(model))

# Later, you can provide this evaluator to the trainer to get results during training

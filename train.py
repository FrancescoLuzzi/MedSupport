from pathlib import Path

import polars as pl
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import (
    SimilarityFunction,
    TripletEvaluator,
)
from sentence_transformers.losses import TripletDistanceMetric, TripletLoss
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)

datasets_folder = Path("./datasets")
dataset_triplets = datasets_folder / "triplets.parquet"

triplets = pl.read_parquet(dataset_triplets)

size = len(triplets)
EVAL_SIZE_PERCENTAGE = 0.20
TEST_SIZE_PERCENTAGE = 0.20
eval_size = int(size * EVAL_SIZE_PERCENTAGE)
test_size = int(size * EVAL_SIZE_PERCENTAGE)
train_size = size - eval_size - test_size

train_dataset = triplets.slice(0, train_size)
eval_dataset = triplets.slice(train_size, eval_size)
test_dataset = triplets.slice(train_size + eval_size)

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

loss = TripletLoss(model, distance_metric=TripletDistanceMetric.COSINE)

# Initialize the evaluator
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"].to_list(),
    positives=eval_dataset["positive"].to_list(),
    negatives=eval_dataset["negative"].to_list(),
    main_similarity_function=SimilarityFunction.COSINE,
    name="med-support-dev",
)
print(dev_evaluator(model))

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/paraphrase-italian-mpnet-med-v2",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if your GPU can't handle FP16
    bf16=False,  # Set to True if your GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="fluzzi/paraphrase-italian-mpnet-med-v2",  # Used in W&B if `wandb` is installed
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=Dataset.from_polars(train_dataset),
    eval_dataset=Dataset.from_polars(eval_dataset),
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"].to_list(),
    positives=test_dataset["positive"].to_list(),
    negatives=test_dataset["negative"].to_list(),
    main_similarity_function=SimilarityFunction.COSINE,
    name="med-support-dev",
)
print(test_evaluator(model))

model.save_pretrained("models/paraphrase-italian-mpnet-med-v2/final")

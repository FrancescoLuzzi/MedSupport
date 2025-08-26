from enum import StrEnum
from itertools import combinations
from pathlib import Path

import polars as pl
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import (
    SimilarityFunction,
    TripletEvaluator,
)
from sentence_transformers.losses import (
    CosineSimilarityLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)


class DataframeColumn(StrEnum):
    PrincipioAttivo = "Principio attivo"
    Farmaco = "Farmaco"
    Confezione = "Confezione"


training_data_folder = Path("./data")
training_dataset_file = training_data_folder / "lista_farmaci_equivalenti.csv"

formats = [
    "compresse",
    "polvere",
    "soluzione orale",
    "iniettabil",
    "infusione",
    "gocce",
    "inalazione",
    "intravenoso",
    "intramuscolare",
]
len_formats = len(formats)

bad_replacement_formats = {"format": [], "replacement": []}

for indx, value in enumerate(combinations(formats, len_formats - 1)):
    bad_replacement_formats["format"].append(formats[len_formats - indx - 1])
    bad_replacement_formats["replacement"].append(value)

bad_replacement_keys = "(" + "|".join(bad_replacement_formats["format"]) + ")"
print(bad_replacement_keys)

df_bad_formats = pl.from_dict(bad_replacement_formats).explode("replacement").lazy()

df = pl.read_csv(training_dataset_file, separator=";")

df = df.select(
    [
        DataframeColumn.PrincipioAttivo,
        DataframeColumn.Farmaco,
        DataframeColumn.Confezione,
    ]
)
print(df)

triplets_query = (
    df.lazy()
    .with_columns(
        format=pl.col(DataframeColumn.Confezione)
        .str.extract_all(bad_replacement_keys)
        .list.unique()
    )
    .filter(pl.col("format").list.len() > 0)
    .explode("format")
    .join(df_bad_formats, on="format")
    .with_columns(
        anchor=pl.col(DataframeColumn.Farmaco) + " in " + pl.col("format"),
        positive=pl.col(DataframeColumn.Farmaco)
        + ": "
        + pl.col(DataframeColumn.Confezione),
        negative=pl.col(DataframeColumn.Farmaco)
        + ": "
        + pl.struct(
            [
                pl.col(DataframeColumn.Confezione),
                pl.col("format"),
                pl.col("replacement"),
            ]
        ).map_elements(
            lambda x: x[DataframeColumn.Confezione].replace(
                x["format"],
                x["replacement"],
            ),
            return_dtype=pl.String,
        ),
    )
    .select("anchor", "positive", "negative")
)

triplets = triplets_query.collect()

EVAL_SIZE_PERCENTAGE = 0.20
TEST_SIZE_PERCENTAGE = 0.20
all_columns = len(triplets.columns)
eval_size = int(all_columns * EVAL_SIZE_PERCENTAGE)
test_size = int(all_columns * EVAL_SIZE_PERCENTAGE)
train_size = all_columns - eval_size - test_size

train_dataset = triplets.slice(0, train_size)
eval_dataset = triplets.slice(train_size, eval_size)
test_dataset = triplets.slice(train_size + eval_size)


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
    run_name="mpnet-base-all-nli-triplet",  # Used in W&B if `wandb` is installed
)
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

loss = MultipleNegativesRankingLoss(model)
# loss = CosineSimilarityLoss(model)

# Initialize the evaluator
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"].to_list(),
    positives=eval_dataset["positive"].to_list(),
    negatives=eval_dataset["negative"].to_list(),
    main_similarity_function=SimilarityFunction.COSINE,
    name="med-support-dev",
)
print(dev_evaluator(model))

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"].to_list(),
    positives=test_dataset["positive"].to_list(),
    negatives=test_dataset["negative"].to_list(),
    name="all-nli-test",
)
test_evaluator(model)

model.save_pretrained("models/mpnet-base-all-nli-triplet/final")

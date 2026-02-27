#!/usr/bin/env python3
import argparse
from pathlib import Path

import polars as pl
from onnxruntime.quantization import quantize_dynamic
from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
)
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

from datasets import Dataset

EVAL_SIZE_PERCENTAGE = 0.20
TEST_SIZE_PERCENTAGE = 0.20


def main():
    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("./datasets"),
        help="Directory containing triplets.parquet",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/paraphrase-italian-mpnet-med-v2"),
        help="Base output model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="ONNX output directory (default: MODEL_DIR/onnx)",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="paraphrase-italian-mpnet-med-v2",
        help="Model name used for output filenames",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="Model to start the finetuning from",
    )
    args = parser.parse_args()

    dataset_triplets = args.dataset_dir / "triplets.parquet"
    model_directory = args.model_dir
    onnx_model_directory = args.output_dir or model_directory / "onnx"
    output_model_name = args.output_model
    onnx_model_file = onnx_model_directory / f"{output_model_name}.onnx"
    onnx_quantized_model_file = (
        onnx_model_directory / f"{output_model_name}_quantized.onnx"
    )

    triplets = pl.read_parquet(dataset_triplets)
    size = len(triplets)
    eval_size = int(size * EVAL_SIZE_PERCENTAGE)
    test_size = int(size * TEST_SIZE_PERCENTAGE)
    train_size = size - eval_size - test_size

    train_dataset = triplets.slice(0, train_size)
    eval_dataset = triplets.slice(train_size, eval_size)
    test_dataset = triplets.slice(train_size + eval_size)

    model = SentenceTransformer(args.base_model)

    loss = TripletLoss(model, distance_metric=TripletDistanceMetric.COSINE)

    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"].to_list(),
        positives=eval_dataset["positive"].to_list(),
        negatives=eval_dataset["negative"].to_list(),
        main_similarity_function=SimilarityFunction.COSINE,
        name=f"evaluate: {output_model_name}",
    )

    print(dev_evaluator(model))

    args_training = SentenceTransformerTrainingArguments(
        output_dir=str(model_directory),
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name=f"training: {output_model_name}",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args_training,
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
        name=f"test: {output_model_name}",
    )
    print(test_evaluator(model))

    model.save_pretrained(model_directory)

    print("Exporting to ONNX...")
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        model_directory,
        export=True,
    )
    ort_model.save_pretrained(onnx_model_directory)
    print(f"ONNX model saved to {onnx_model_directory}")

    print("Generating quantized ONNX model...")
    quantize_dynamic(onnx_model_file, onnx_quantized_model_file)


if __name__ == "__main__":
    main()

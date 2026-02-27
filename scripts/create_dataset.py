#!/usr/bin/env python3
import argparse
from enum import StrEnum
from itertools import combinations
from pathlib import Path

import polars as pl


class DataframeColumn(StrEnum):
    PrincipioAttivo = "Principio attivo"
    Farmaco = "Farmaco"
    Confezione = "Confezione"


FORMATS = [
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


def main():
    parser = argparse.ArgumentParser(prog="create_dataset.py")
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("./data"),
        help="Source directory for CSV files",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./models"),
        help="Model directory (unused)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./datasets"),
        help="Output directory for triplets dataset",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Model name (unused)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=69420,
        help="Randomizing seed",
    )

    args = parser.parse_args()

    training_data_folder = args.src_dir
    training_dataset_file = training_data_folder / "lista_farmaci_equivalenti.csv"
    datasets_folder = args.output_dir
    dataset_triplets = datasets_folder / "triplets.parquet"
    seed = args.seed

    len_formats = len(FORMATS)

    bad_replacement_formats = {"format": [], "replacement": []}
    for indx, value in enumerate(combinations(FORMATS, len_formats - 1)):
        bad_replacement_formats["format"].append(FORMATS[len_formats - indx - 1])
        bad_replacement_formats["replacement"].append(value)

    bad_replacement_keys = "(" + "|".join(bad_replacement_formats["format"]) + ")"

    df_bad_formats = pl.from_dict(bad_replacement_formats).explode("replacement").lazy()

    df = pl.read_csv(training_dataset_file, separator=";").select(
        [
            DataframeColumn.PrincipioAttivo,
            DataframeColumn.Farmaco,
            DataframeColumn.Confezione,
        ]
    )

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
        .with_columns(pl.col("negative").sample(fraction=1.0, shuffle=True, seed=seed))
    )

    triplets_query.collect().write_parquet(dataset_triplets)


if __name__ == "__main__":
    main()

from enum import StrEnum
from itertools import combinations
from pathlib import Path

import polars as pl


class DataframeColumn(StrEnum):
    PrincipioAttivo = "Principio attivo"
    Farmaco = "Farmaco"
    Confezione = "Confezione"


training_data_folder = Path("./data")
training_dataset_file = training_data_folder / "lista_farmaci_equivalenti.csv"
datasets_folder = Path("./datasets")
dataset_triplets = datasets_folder / "triplets.parquet"

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

df_bad_formats = pl.from_dict(bad_replacement_formats).explode("replacement").lazy()

df = pl.read_csv(training_dataset_file, separator=";").select(
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

triplets_query.collect().write_parquet(dataset_triplets)

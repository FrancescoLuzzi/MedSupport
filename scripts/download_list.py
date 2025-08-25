#!/usr/bin/env python3
import argparse
import functools
from collections.abc import Callable
from os import path
from pathlib import Path

import requests
from tqdm.auto import tqdm

_filename = path.basename(__file__)
parser = argparse.ArgumentParser(prog=_filename)
parser.add_argument(
    "data_dir",
    type=Path,
    default="./data",
    nargs="?",
)

args = parser.parse_args()


def download_file(
    url: str,
    destination_path: str | Path,
    columns_hint: int = 0,
    line_manipulation: Callable[[int, bytes], bytes] | None = None,
    chunk_size: int = 16384,  # 4KB
):
    r = requests.get(
        url,
        stream=True,
        allow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
        },
    )
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))
    path = Path(destination_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = path.name + "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(
        r.raw.read,
        decode_content=True,  # Decompress if needed
    )
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            chunk = r_raw.read(chunk_size)
            remained_chunk = b""
            line_number = 0
            semicolons_in_header = columns_hint - 1
            while chunk:
                chunk = remained_chunk + chunk
                remained_chunk = b""
                chunk = chunk.replace(b"\r", b"").replace(b" \x80", b"")
                for line in chunk.splitlines():
                    line = remained_chunk + line
                    semicolons = line.count(b";")
                    if semicolons_in_header == -1:
                        semicolons_in_header = semicolons
                    if line.count(b";") == semicolons_in_header:
                        if line_manipulation:
                            line = line_manipulation(line_number, line)
                        remained_chunk = b""
                        line_number += 1
                        f.write(line + b"\n")
                    else:
                        remained_chunk = line
                chunk = r_raw.read(chunk_size)


destination_data_dir: Path = args.data_dir

lista_farmaci_equivalenti = destination_data_dir / "lista_farmaci_equivalenti.csv"
anagrafica_farmaci = destination_data_dir / "anagrafica_farmaci.csv"
principi_attivi = destination_data_dir / "principi_attivi.csv"
anagrafica_atc = destination_data_dir / "anagrafica_atc.csv"


def create_maintain_only_bytes_in_column_header_handler(
    chunk_to_maintain: bytes,
) -> Callable[[int, bytes], bytes]:
    """
    Returns an handler that:
    - maintains the specified bytes in a column header
    - decoded each line from latin1 and encodes them as utf8
    - lower cases all the csv lines different from the header

    Example:
        column_headers = "My;Simple;Example to remove;Last"
        handler = create_maintain_only_bytes_in_column_header_handler(b"example")
        assert handler(0,column_headers) == "My;Simple;Example;Last"
        assert handler(1,column_headers) == "my;simple;example to remove;last"
    """
    chunk_to_maintain_len = len(chunk_to_maintain)

    def handler(line_number: int, chunk: bytes) -> bytes:
        if line_number == 0:
            indx = chunk.find(chunk_to_maintain)
            while indx >= 0:
                start = indx + chunk_to_maintain_len
                end = chunk.find(b";", start)
                chunk = chunk[:start] + chunk[end:]
                indx = chunk.find(chunk_to_maintain)

        str_chunk = chunk.decode("latin1")
        if line_number != 0:
            str_chunk = str_chunk.lower()

        return str_chunk.encode("utf8")

    return handler


download_file(
    "https://www.aifa.gov.it/documents/20142/825643/Lista_farmaci_equivalenti.csv",
    lista_farmaci_equivalenti,
    columns_hint=12,
    line_manipulation=create_maintain_only_bytes_in_column_header_handler(
        b"Prezzo Pubblico"
    ),
)

download_file("https://drive.aifa.gov.it/farmaci/confezioni.csv", anagrafica_farmaci)
download_file("https://drive.aifa.gov.it/farmaci/PA_confezioni.csv", principi_attivi)
download_file("https://drive.aifa.gov.it/farmaci/atc.csv", anagrafica_atc)

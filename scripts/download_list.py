#!/usr/bin/env python3
import argparse
import asyncio
import re
from collections.abc import Callable
from pathlib import Path

import aiohttp
from aiohttp import ClientTimeout
from tqdm.asyncio import tqdm



async def download_file(
    url: str,
    destination_path: Path,
    pos: int,
    *,
    columns_hint: int = 0,
    line_manipulation: Callable[[int, bytes], bytes] | None = None,
    chunk_size: int = 4096,
):
    timeout = ClientTimeout(total=30, sock_read=10)
    destination_path = destination_path.expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, allow_redirects=True) as response:
            if response.status != 200:
                response.raise_for_status()
            file_size = int(response.headers.get("Content-Length", 0))
            desc = (
                destination_path.name
                if file_size > 0
                else f"{destination_path.name} (Unknown)"
            )

            with (
                destination_path.open("wb") as f,
                tqdm(
                    total=file_size if file_size > 0 else None,
                    desc=desc,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    position=pos,
                    leave=False,
                ) as pbar,
            ):
                remained_chunk = b""
                line_number = 0
                semicolons_in_header = columns_hint - 1

                async for chunk in response.content.iter_chunked(chunk_size):
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

                    pbar.update(len(chunk))
                # indicate download completion in console
                pbar.write(f"Downloaded {destination_path.name}")

    return destination_path


def create_maintain_only_bytes_in_column_header_handler(
    chunk_to_maintain: bytes,
) -> Callable[[int, bytes], bytes]:
    pattern = re.compile(re.escape(chunk_to_maintain) + b"[^;]*")

    def handler(line_number: int, chunk: bytes) -> bytes:

        if line_number != 0:
            return chunk.decode("latin1").lower().encode("utf-8")
        else:
            chunk = pattern.sub(chunk_to_maintain, chunk)
            return chunk.decode("latin1").encode("utf-8")

    return handler


async def main():
    parser = argparse.ArgumentParser(prog="download_list.py")
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("."),
        help="Source directory (unused)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Model directory (unused)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Destination directory for downloaded CSV files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Model name (unused)",
    )
    args = parser.parse_args()
    destination_data_dir: Path = args.output_dir

    lista_farmaci_equivalenti = destination_data_dir / "lista_farmaci_equivalenti.csv"
    anagrafica_farmaci = destination_data_dir / "anagrafica_farmaci.csv"
    principi_attivi = destination_data_dir / "principi_attivi.csv"
    anagrafica_atc = destination_data_dir / "anagrafica_atc.csv"

    tasks = [
        download_file(
            "https://drive.aifa.gov.it/farmaci/confezioni.csv", anagrafica_farmaci, 0
        ),
        download_file(
            "https://drive.aifa.gov.it/farmaci/PA_confezioni.csv", principi_attivi, 1
        ),
        download_file("https://drive.aifa.gov.it/farmaci/atc.csv", anagrafica_atc, 2),
        download_file(
            "https://www.aifa.gov.it/documents/20142/825643/Lista_farmaci_equivalenti.csv",
            lista_farmaci_equivalenti,
            3,
            columns_hint=12,
            line_manipulation=create_maintain_only_bytes_in_column_header_handler(
                b"Prezzo Pubblico"
            ),
        ),
    ]

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())

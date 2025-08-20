#! /usr/bin/bash

destination_dir=${1:-./data}

mkdir -p "$destination_dir" 2>&1 >/dev/null

curl "https://www.aifa.gov.it/documents/20142/825643/Lista_farmaci_equivalenti.csv" | sed -e "s/ *\x80//g" -e "s/\r//g" -e "s/;Prezzo Pubblico.*;/;Prezzo Pubblico;/" > "$destination_dir/lista_farmaci_equivalenti.csv"

curl "https://drive.aifa.gov.it/farmaci/confezioni.csv" -o "$destination_dir/anagrafica_farmaci.csv"
curl "https://drive.aifa.gov.it/farmaci/PA_confezioni.csv" -o "$destination_dir/principi_attivi.csv"
curl "https://drive.aifa.gov.it/farmaci/atc.csv" -o "$destination_dir/anagrafica_atc.csv"

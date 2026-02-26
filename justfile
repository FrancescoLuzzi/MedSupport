default:
    @just --list

download:
    @python scripts/download_list.py

train:
    @python train.py

run:
    @python main.py

dataset:
    @python create_dataset.py

serve-test:
    @npx http-server -p 8080 --cors -c-1

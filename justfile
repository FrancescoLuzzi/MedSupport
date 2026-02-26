default:
    @just --list

train:
    @python train.py

run:
    @python main.py

dataset:
    @python create_dataset.py

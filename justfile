default:
    @just --list

train:
    @python3 train.py

dataset:
    @python3 create_dataset.py

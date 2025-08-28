default:
    @just --list

train:
    @python3 train.py

export-onnx source destination:
    @optimum-cli export onnx -m {{source}} --task feature-extraction {{destination}}

dataset:
    @python3 create_dataset.py

default:
    @just --list

train:
    @python train.py

run:
    @python main.py

dataset:
    @python create_dataset.py

# Serve ONNX test page (open http://localhost:8080/test_onnx.html)
serve-test:
    npx http-server -p 8080 --cors -c-1

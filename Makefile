.PHONY: install setup-yolov5 test lint prepare train

install:
	python -m pip install -e ".[dev]"

setup-yolov5:
	python scripts/bootstrap_yolov5.py
	python -m pip install -r third_party/yolov5/requirements.txt

test:
	pytest -q

lint:
	ruff check .

prepare:
	python scripts/prepare_dataset.py --images-dir data/raw/images --labels-dir data/raw/labels --output-dir data/processed --seed 42

train:
	python scripts/train.py --yolov5-dir third_party/yolov5 --data configs/building_disease.yaml


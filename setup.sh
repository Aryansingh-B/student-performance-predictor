#!/bin/bash
mkdir -p model
python data/generate_data.py
python -m src.train
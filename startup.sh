#!/bin/bash
if [ ! -f "model/model.pkl" ]; then
    echo "Model not found. Training now..."
    python data/generate_data.py
    python -m src.train
fi
streamlit run app/streamlit_app.py
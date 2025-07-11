# ECG-Based Continual Learning Framework with Interpretability and Drift Detection

## Overview

This project implements an ECG classification pipeline that integrates Transformer-based sequence modeling, continual learning strategies, interpretability techniques, and drift detection mechanisms. The goal is to simulate real-world deployment scenarios where data distributions shift over time, such as across different age groups.

## Project Structure

```
.
├── data_preprocessing/
│   ├── data_processing.py
│   ├── process_dm_data.py
│   ├── process_missing_values.py
│   └── process_additional_features.py
├── baselines/
│   ├── cnn_baseline_model.py
│   └── lstm_baseline_model.py
├── transformer/
│   ├── new_train_transformer.py
│   ├── new_transformer_feature_extractor.py
│   └── new_classifier.py
├── explainability/
│   ├── shap_explain.py
│   ├── feature_analysis.py
│   └── attention_heatmap_visuals.py
├── drift/
│   ├── mmd_detector.py
│   ├── drift_detection_module.py
│   └── drift_visualization.py
├── continual_learning/
│   ├── ewc.py
│   ├── replay.py
│   ├── ranpac.py
│   ├── gem.py
│   ├── lwf.py
│   └── icarl.py
├── evaluation/
│   ├── evaluation.py
│   └── statistical_analysis.py
└── main.py
```

## Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:

* numpy, pandas, scikit-learn
* torch, dgl
* matplotlib, seaborn
* shap, scipy

## How to Run

1. Preprocess data

   ```bash
   python data_processing.py
   ```

2. Train the Transformer model

   ```bash
   python new_train_transformer.py
   ```

3. Evaluate or interpret results

   ```bash
   python shap_explain.py
   python drift_visualization.py
   python evaluation.py
   ```

4. Execute continual learning experiments

   ```bash
   python main.py
   ```

## Summary of Results

* The Replay strategy consistently achieves the best performance across tasks.
* RanPAC performs competitively while enhancing interpretability through randomized attention control.
* GEM and iCaRL suffer from significant forgetting under domain shift.
* Performance is evaluated through accuracy, F1 score, recall, AUC, and visualized using violin plots, line charts, and attention/SHAP maps.



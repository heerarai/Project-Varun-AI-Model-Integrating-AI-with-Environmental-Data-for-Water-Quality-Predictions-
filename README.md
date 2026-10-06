# Project Varuna
Author: Suhani Rai, Dristi Sharma, Harshini Chandilvel

## Explainable AI for Water Quality Prediction

---

## Overview

Project Varuna is an explainable artificial intelligence (AI) system designed to predict water quality conditions and generate human-readable explanations from measured water-quality and environmental data.
Traditional water quality analysis methods are often slow, labor-intensive, and difficult for non-experts to interpret. Varuna demonstrates how machine learning and large language models can be combined
to provide faster, interpretable decision support for environmental monitoring. This project focuses on water quality data from the Austin, Texas region and serves as a research-grade 
prototype rather than a certified safety assessment tool.

---

## Key Features

- End-to-end data pipeline from raw datasets to predictions
- Water Quality Index (WQI)–based labeling using scientific thresholds
- Baseline Random Forest model for accurate classification and feature importance analysis
- Explainability layer grounded in physicochemical drivers
- Fine-tuned FLAN-T5 language model for natural-language explanations
- Interactive web-based demo for real-time prediction and interpretation

---

## Datasets

All data used in this project is publicly available.

- Water Quality Data:
  - Austin-area water quality sampling data containing measurements such as pH, dissolved oxygen, turbidity, nitrate, phosphorus, and water temperature.
- Environmental Data:
  - Daily environmental data (e.g., daylight duration, temperature) obtained from an open-access weather data source.

No physical samples, chemicals, or biological materials were used.

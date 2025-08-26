# Multilingual Natural Language Inference with DeBERTa

This project implements a solution for the Natural Language Inference (NLI) task using a multilingual DeBERTa model. The system classifies logical relationships between sentence pairs in multiple languages into three categories: entailment, contradiction, or neutral.

Kaggle hackathon: https://www.kaggle.com/competitions/contradictory-my-dear-watson

## Features

- **Multilingual Support**: Based on mDeBERTa-v3-base model pretrained on XNLI dataset
- **Custom Architecture**: Enhanced with additional dense layers for better classification performance
- **Efficient Training**: Partial layer freezing and optimized training pipeline
- **Batch Processing**: Memory-efficient inference for large datasets
- **Compatible Output**: Generates submission files in Kaggle competition format

## Project Structure

```
/
├── best_model/                 # Saved model and tokenizer
│   ├── config.json            # Model configuration
│   ├── model.safetensors      # Model weights
│   └── tokenizer files        # Tokenizer configuration
├── submission.csv             # Generated predictions
├── train.py                   # Main training script
└── requirements.txt           # Python dependencies
```

## Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/ksnnchk/kaggle-3-contradictory.git
cd kaggle-3-contradictory
```

2. **Install dependencies**
```bash
pip install torch transformers pandas scikit-learn numpy
```

Key dependencies:
- torch>=2.0.0
- transformers>=4.30.0
- pandas>=1.5.0
- scikit-learn>=1.2.0
- numpy>=1.22.0

## Dataset

The project uses the "Contradictory, My Dear Watson" dataset from Kaggle, containing:

- **Premise-Hypothesis pairs** in multiple languages
- **Three labels**: 0 (entailment), 1 (neutral), 2 (contradiction)
- **Training set**: Labeled examples for model training
- **Test set**: Unlabeled examples for prediction submission

## Model Architecture

### CustomDeBERTa Class
```python
CustomDeBERTa(
    model_name="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    num_labels=3
)
```

**Key components:**
- **Base Model**: Frozen mDeBERTa layers (except last 4)
- **Classifier Head**: 
  - Linear(768 → 512) + ReLU + Dropout + LayerNorm
  - Linear(512 → 256) + GELU + Dropout
  - Linear(256 → 3) → Output logits

## Training Configuration

```python
TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    eval_strategy="epoch",
    metric_for_best_model="accuracy"
)
```

**Training Features:**
- Early stopping based on validation accuracy
- Gradient checkpointing for memory efficiency
- Automatic mixed precision (AMP) support
- Model checkpointing

## Performance

- **Accuracy**: >88% on validation set

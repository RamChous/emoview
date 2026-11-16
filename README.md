# EmoView

EmoView classifies English tweets into six emotions: sadness, joy, love, anger, fear, and surprise. This Applied Machine Learning final project compares a TF-IDF and Logistic Regression baseline with fine-tuned DistilBERT.

## Results

Both models were evaluated on the same held-out test set of 37,153 tweets. DistilBERT was selected using validation weighted F1.

| Model | Test accuracy | Weighted F1 | Macro F1 |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 96.95% | 0.9696 | 0.9614 |
| Fine-tuned DistilBERT | **99.58%** | **0.9958** | **0.9937** |

Surprise had the lowest per-class F1 for DistilBERT (0.9869). The notebook includes per-class precision and recall, confusion matrices, and examples of high-confidence errors.

## Dataset and method

The [DAIR-AI Emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) (unsplit configuration) contains 416,809 labeled English tweets. The notebook normalizes text to identify repeated messages, removes groups with conflicting labels, and keeps one copy of each remaining repeated message. The resulting 371,527 examples are split into training, validation, and test sets with an 80/10/10 stratified split (seed 42). Each normalized text appears in only one split.

- **TF-IDF + Logistic Regression** uses word unigrams and bigrams, up to 50,000 features, and balanced class weights.
- **DistilBERT** starts from `distilbert-base-uncased` and is fine-tuned for three epochs with a maximum input length of 128 tokens.

## Run the notebook

1. Open [EmoView_AML_Final_Project(5).ipynb](EmoView_AML_Final_Project%285%29.ipynb) in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**. The notebook enables FP16 for DistilBERT training.
3. Run all cells in order. The notebook installs its Hugging Face dependencies and downloads the dataset and pretrained checkpoint.

The notebook saves plots and training checkpoints in `emoview_dair_outputs/`. It saves the selected DistilBERT model and tokenizer in `emoview_model/` and packages them as `emoview_model.zip` for reuse.

## Limitations

The reported scores apply to the cleaned Twitter dataset. Automatically obtained labels can be noisy, and near-duplicate messages may still appear across splits. Performance on customer reviews or sarcastic text requires separate evaluation. Neutral and frustration are outside the six-label scheme.

## Authors

Rakhmettula Ramazan and Agzam Sayat

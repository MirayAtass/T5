import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import pipeline
import torch
from transformers import Trainer
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score

train_df = pd.read_csv("train_balanced.csv")
val_df = pd.read_csv("val_balanced.csv")
test_df = pd.read_csv("test_balanced.csv")

print("Eğitim verisi örneği:\n", train_df.head(3))
print("Doğrulama verisi örneği:\n", val_df.head(3))
print("Test verisi örneği:\n", test_df.head(3))

le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['label'])
val_df['label'] = le.transform(val_df['label'])
test_df['label'] = le.transform(test_df['label'])

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

print("Label mapping (class names to integers):", dict(zip(le.classes_, le.transform(le.classes_))))
id2label = {i: label for i, label in enumerate(le.classes_)}
train_df['label_text'] = train_df['label'].map(id2label).astype(str)
val_df['label_text'] = val_df['label'].map(id2label).astype(str)
test_df['label_text'] = test_df['label'].map(id2label).astype(str)

train_dataset = Dataset.from_pandas(train_df[['text', 'label_text']], preserve_index=False)
val_dataset = Dataset.from_pandas(val_df[['text', 'label_text']], preserve_index=False)
test_dataset = Dataset.from_pandas(test_df[['text', 'label_text']], preserve_index=False)

example = train_dataset[0]
input_text = example["text"]
target_text = example["label_text"]

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

input_enc = tokenizer(
    input_text,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

target_enc = tokenizer(
    target_text,
    padding="max_length",
    truncation=True,
    max_length=10,
    return_tensors="pt"
)

print("Input text:", input_text)
print("Input IDs:", input_enc.input_ids)
print("Target text:", target_text)
print("Target IDs:", target_enc.input_ids)

def preprocess_function(examples):
    inputs = ["sentiment: " + text for text in examples["text"]]
    model_inputs = tokenizer(
        inputs, max_length=128, truncation=True, padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["label_text"], max_length=10, truncation=True, padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tokenized = train_dataset.map(preprocess_function, batched=True)
val_tokenized = val_dataset.map(preprocess_function, batched=True)
test_tokenized = test_dataset.map(preprocess_function, batched=True)

print("Tokenize edilmiş eğitim örneği:\n", train_tokenized[0])

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)

predictions = []
true_labels = [example["label_text"] for example in test_dataset]

for example in tqdm(test_dataset):
    input_text = example["text"]
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    ).to(model.device)

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=10
    )

    pred_label = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predictions.append(pred_label)

for i in range(5):
    print(f"Text: {test_dataset[i]['text']}")
    print(f"True label: {true_labels[i]}")
    print(f"Predicted label: {predictions[i]}")
    print("-" * 40)

results = trainer.evaluate(eval_dataset=test_tokenized)
print("\n--- Test Sonuçları ---")
print(results)

preds_output = trainer.predict(test_tokenized)

pred_ids = preds_output.predictions
if isinstance(pred_ids[0][0], (list, np.ndarray)):
    pred_ids = [p[0] for p in pred_ids]

decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(preds_output.label_ids, skip_special_tokens=True)

label_map = {'Negative': 0, 'Notr': 1, 'Positive': 2}
y_true = [label_map[label] for label in decoded_labels]
y_pred = [label_map.get(pred, 1) for pred in decoded_preds]
print("\n--- Detaylı Test Raporu ---")
print(classification_report(y_true, y_pred, target_names=['Negative', 'Notr', 'Positive']))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nMacro F1 Score:", f1_score(y_true, y_pred, average='macro'))

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Notr', 'Positive'])
cm_display.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("t5_confusion_matrix.png", dpi=300)
plt.show()

report = classification_report(y_true, y_pred, target_names=['Negative', 'Notr', 'Positive'], output_dict=True)
metrics_df = pd.DataFrame(report).T.iloc[:3][['precision', 'recall', 'f1-score']]

metrics_df.plot(kind='bar', figsize=(8, 6), colormap='viridis')
plt.title('Classification Metrics by Class')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("t5_classification_report_metrics.png", dpi=300)
plt.show()
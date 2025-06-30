# ===================================
# SIMPLE WORKING VIT-CODEGPT SETUP
# ===================================

import os
import multiprocessing as mp
import pandas as pd
import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import (
    VisionEncoderDecoderModel, 
    ViTFeatureExtractor, 
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback
)
from huggingface_hub import login
import evaluate
import warnings
warnings.filterwarnings("ignore")

# ===================================
# 1. SETUP & LOGIN
# ===================================

#HUGGINGFACE_HUB_TOKEN = ""
#login(token=HUGGINGFACE_HUB_TOKEN)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["WANDB_DISABLED"] = "true"

# ===================================
# 2. CONFIGURATION
# ===================================

class Config:
    ENCODER = "google/vit-base-patch16-224"
    DECODER = "microsoft/CodeGPT-small-py"
    
    TRAIN_BATCH_SIZE = 4
    VAL_BATCH_SIZE = 4
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 3
    MAX_LENGTH = 256
    IMG_SIZE = (224, 224)
    
    GRADIENT_ACCUMULATION_STEPS = 4
    FP16 = True
    EVAL_STEPS = 500
    SAVE_STEPS = 500
    LOGGING_STEPS = 100
    SAVE_TOTAL_LIMIT = 3
    
    HUB_MODEL_ID = "Thehunter99/vit-codegpt-cadcoder"
    SEED = 42

config = Config()

# ===================================
# 3. MODEL SETUP (FIXED)
# ===================================

print("🚀 Setting up model and tokenizer...")

feature_extractor = ViTFeatureExtractor.from_pretrained(config.ENCODER)
tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Tokenizer loaded. Vocab size: {len(tokenizer)}")

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    config.ENCODER, 
    config.DECODER
)

# FIXED Configuration
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Generation config (NO beam search during training)
model.config.max_length = config.MAX_LENGTH
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 1.0
model.config.num_beams = 1
model.config.do_sample = False

model.gradient_checkpointing_enable()
print("✅ Model setup complete!")

# ===================================
# 4. DATA LOADING
# ===================================

print("📁 Loading data...")

cadquery_dir = "CADCODER_GenCAD-Code_download/cadquery"
images_dir = "CADCODER_GenCAD-Code_download/images"

cadquery_texts = []
file_ids = []

cadquery_files = sorted([f for f in os.listdir(cadquery_dir) if f.endswith('.txt')])

# Load first 5000 samples for faster training
for filename in cadquery_files[:10000]:
    filepath = os.path.join(cadquery_dir, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if len(content) > 50:
                cadquery_texts.append(content)
                file_id = filename.replace('cadquery_', '').replace('.txt', '')
                file_ids.append(f"image_{file_id}.png")
    except Exception as e:
        continue

df = pd.DataFrame({'image': file_ids, 'caption': cadquery_texts})
print(f"📊 Loaded {len(df)} samples")

train_df, val_df = train_test_split(df, test_size=0.15, random_state=config.SEED)
print(f"🔄 Train: {len(train_df)}, Val: {len(val_df)}")

# ===================================
# 5. DATASET CLASS
# ===================================

class SimpleCADDataset(Dataset):
    def __init__(self, df, root_dir, tokenizer, feature_extractor, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.max_length = config.MAX_LENGTH
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            caption = self.df.caption.iloc[idx]
            image_name = self.df.image.iloc[idx]
            img_path = os.path.join(self.root_dir, image_name)
            
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            
            pixel_values = self.feature_extractor(
                img, return_tensors="pt", do_rescale=False
            ).pixel_values.squeeze()
            
            encoded = self.tokenizer(
                caption,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded.input_ids.squeeze()
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {"pixel_values": pixel_values, "labels": labels}
            
        except Exception as e:
            return {
                "pixel_values": torch.zeros((3, 224, 224)),
                "labels": torch.full((self.max_length,), -100, dtype=torch.long)
            }

# ===================================
# 6. METRICS
# ===================================

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    try:
        predictions, labels = eval_pred
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        rouge_result = rouge.compute(
            predictions=decoded_preds, 
            references=decoded_labels,
            rouge_types=["rouge1", "rouge2", "rougeL"]
        )
        
        return {
            "rouge1": round(rouge_result["rouge1"], 4),
            "rouge2": round(rouge_result["rouge2"], 4),
            "rougeL": round(rouge_result["rougeL"], 4),
        }
        
    except Exception as e:
        print(f"❌ Metrics failed: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

# ===================================
# 7. DATASETS & TRAINING
# ===================================

transform = T.Compose([T.Resize(config.IMG_SIZE), T.ToTensor()])

train_dataset = SimpleCADDataset(train_df, images_dir, tokenizer, feature_extractor, transform)
val_dataset = SimpleCADDataset(val_df, images_dir, tokenizer, feature_extractor, transform)

print(f"📊 Train dataset: {len(train_dataset)}")
print(f"📊 Val dataset: {len(val_dataset)}")

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./vit-codegpt-cadcoder',
    hub_model_id=config.HUB_MODEL_ID,
    push_to_hub=True,
    hub_strategy="checkpoint",
    
    num_train_epochs=config.EPOCHS,
    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.VAL_BATCH_SIZE,
    gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
    learning_rate=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    warmup_ratio=0.1,
    
    fp16=config.FP16,
    dataloader_pin_memory=False,
    
    eval_strategy="steps",
    eval_steps=config.EVAL_STEPS,
    save_strategy="steps",
    save_steps=config.SAVE_STEPS,
    save_total_limit=config.SAVE_TOTAL_LIMIT,
    load_best_model_at_end=True,
    metric_for_best_model="eval_rouge2",
    greater_is_better=True,
    
    logging_strategy="steps",
    logging_steps=config.LOGGING_STEPS,
    
    predict_with_generate=True,
    generation_max_length=config.MAX_LENGTH,
    generation_num_beams=1,
    
    do_train=True,
    do_eval=True,
    overwrite_output_dir=True,
    report_to=[],
    seed=config.SEED,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

# ===================================
# 8. TRAINING
# ===================================

print("🔥 Starting training...")
trainer.train()

print("💾 Saving final model...")
trainer.save_model("./final-model")
trainer.push_to_hub(commit_message="Final trained VIT-CodeGPT model")

print("📊 Final evaluation...")
eval_results = trainer.evaluate()
print("Final Results:", eval_results)

print("✅ Training completed!")
print(f"🌐 Model available at: https://huggingface.co/{config.HUB_MODEL_ID}")

# ===================================
# 9. TEST INFERENCE
# ===================================

def test_model():
    """Test the trained model"""
    print("🧪 Testing inference...")
    
    test_image_path = "CADCODER_GenCAD-Code_download/images/image_000000.png"
    
    if os.path.exists(test_image_path):
        image = Image.open(test_image_path)
        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
        
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=256,
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("🎯 Generated CAD Code:")
        print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
    else:
        print("❌ Test image not found")

#test after training:
test_model()
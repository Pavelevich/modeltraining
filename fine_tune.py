import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class QADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length=512):
        logging.info("Initializing QADataset.")
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        pair = self.qa_pairs[idx]
        prompt = f"User: {pair['answer']}\nAssistant:"
        response = f" {pair['question']}"

        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        labels = self.tokenizer(
            response,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        ).input_ids

        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()
        labels = labels.squeeze()

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def load_model():
    model_path = 'C:/Users/devel/.llama/checkpoints/Llama3.2-1B'
    token = "hf_AbAOGDxpIewOrvijZISAnYBFmsb"  # ⚠️ **Token ⚠️**

    try:
        logging.info(f"Loading the tokenizer for the model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        logging.info("Tokenizer loaded successfully.")

        logging.info(f"Loading the model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # load_in_8bit=True,  # Removed
            # device_map='auto',  # Removed
            use_auth_token=token,  # ⚠️ **Token  ⚠️**
            torch_dtype=torch.float32
        )
        model.to('cpu')
        logging.info("Model loaded successfully and moved to CPU.")

        logging.info("Configuring LoRA for the model.")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        logging.info("LoRA configuration set.")

        logging.info("Applying LoRA to the model.")
        model = get_peft_model(model, peft_config)
        logging.info("LoRA applied to the model successfully.")

        return tokenizer, model

    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        raise

def prepare_dataset(qa_pairs, tokenizer, max_length=512):
    logging.info("Creating the dataset for training.")
    dataset = QADataset(qa_pairs, tokenizer, max_length)
    logging.info("Dataset created successfully.")
    return dataset

def main():
    try:
        logging.info("Loading QA pairs from 'qa_pairs.json'.")
        with open('qa_pairs.json', 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        logging.info(f"Total QA pairs loaded: {len(qa_pairs)}")

        tokenizer, model = load_model()

        logging.info("Preparing the dataset for training.")
        dataset = prepare_dataset(qa_pairs, tokenizer, max_length=512)
        logging.info("Dataset prepared successfully.")

        logging.info("Setting up training arguments.")
        training_args = TrainingArguments(
            output_dir='./finetuned_model',
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=1000,
            save_total_limit=2,
            fp16=True,
        )
        logging.info("Training arguments set up.")

        logging.info("Creating the Trainer.")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        logging.info("Trainer created successfully.")

        logging.info("Starting training.")
        trainer.train()
        logging.info("Training completed.")

        logging.info("Saving the fine-tuned model.")
        trainer.save_model('./finetuned_model')
        tokenizer.save_pretrained('./finetuned_model')
        logging.info("Model and tokenizer saved successfully in './finetuned_model'.")

    except Exception as e:
        logging.error(f"Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()

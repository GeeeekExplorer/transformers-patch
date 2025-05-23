import transformers_patch
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer


class RandomDataset(Dataset):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
    def __len__(self):
        return 1000
    def __getitem__(self, index):
        return {"input_ids": torch.randint(0, 10000, (self.seq_len,)).tolist()}


seq_len = 2048
model_path = "/YOUR/PATH/Qwen3-8B"

rank = int(os.getenv("LOCAL_RANK", "0"))
dataset = RandomDataset(seq_len)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=f"cuda:{rank}")

args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    save_strategy="no",
    bf16=True,
    deepspeed={
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_optimization": {
            "stage": 1
        }
    },
    gradient_checkpointing=False,
    report_to="none",
    max_seq_length=seq_len,
    dataset_kwargs={"skip_prepare_dataset": True},
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=args,
)
trainer.train()
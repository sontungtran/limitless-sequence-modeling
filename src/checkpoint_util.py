"""
For checkpointing while training

Author: 
"""

import os
from os.path import isfile, isdir, join
import re
import json

def save_checkpoint(model, tokenizer,
                    model_dir, name, current_config,
                    epoch, idx,
                    train_loss, val_loss, train_acc, val_acc, train_time, valid_time,
                    nb_tr_steps, nb_tr_examples,
                    other_info=None):
    # Save current epoch #, train loss & acc into a json file
    files = []
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Creating output directory {model_dir}")
    else:    
        files = [f for f in os.listdir(model_dir) if isfile(join(model_dir, f))]
        dirs = [f for f in os.listdir(model_dir) if isdir(join(model_dir, f))]

    # Find checkpoint
    checkpoint_filename = f"ckpt-{name}.json"
    if checkpoint_filename in files:
        with open(os.path.join(model_dir, checkpoint_filename), 'r') as f:
            ckpt_info = json.load(f)
            assert ckpt_info["config"] == current_config # Model config should be the same
    else:
        ckpt_info = {"config": current_config, "info": []}
    print(f"Length of current stats file: {len(ckpt_info['info'])}\n")

    if len(ckpt_info["info"]) > 0:
        assert ckpt_info["info"][-1]["epoch"] <= epoch, f"Mismatch epoch number in checkpoint info file: \"ckpt-{name}.json\": Current epoch: {epoch}, but current epoch should be >={ckpt_info['info'][-1]['epoch']}"
        assert ckpt_info["info"][-1]["idx"] <= idx, f"Mismatch idx number in checkpoint info file: \"ckpt-{name}.json\": Current epoch: {epoch}, but current epoch should be >={ckpt_info['info'][-1]['idx']}"

    # Update 
    ckpt_info["info"].append(
        {
            'epoch': epoch,
            'idx': idx,
            'Training Loss': float(train_loss),
            'Valid. Loss': float(val_loss),
            'Training Acc': float(train_acc),
            'Valid. Acc': float(val_acc),
            'Training Time': float(train_time),
            'Valid. Time': float(valid_time),
            'Num Tr Steps': int(nb_tr_steps),
            'Num Tr Examples': int(nb_tr_examples),
            "Other": other_info
        }
    )

    # Write back into file
    with open(os.path.join(model_dir, checkpoint_filename), 'w') as f:
        json.dump(ckpt_info, f)

    tokenizer.save_pretrained(join(model_dir, f"{name}-{epoch}"))
    model.save_pretrained(join(model_dir, f"{name}-{epoch}"))
    return None


def load_checkpoint(model_dir, name, current_config, model_config):
    # Save current epoch #, train loss & acc into a json file
    files = []
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Creating output directory {model_dir}")
    else:    
        files = [f for f in os.listdir(model_dir) if isfile(join(model_dir, f))]
        dirs = [f for f in os.listdir(model_dir) if isdir(join(model_dir, f))]

    # Find checkpoint
    checkpoint_filename = f"ckpt-{name}.json"
    if checkpoint_filename in files:
        # TODO: If can't load, create new ckpt?
        with open(os.path.join(model_dir, checkpoint_filename), 'r') as f:
            ckpt_info = json.load(f)
            assert ckpt_info["config"] == current_config # Model config should be the same but commented out for now
    else:
#         raise Exception(f"Can't find checkpoint: \"{checkpoint_filename}\"")
        print(f"Can't find checkpoint: \"{checkpoint_filename}\", using downloaded pretrained model")
        tokenizer = RobertaTokenizer.from_pretrained(current_config['model'])
        model = RobertaForSequenceClassification.from_pretrained(current_config['model'], config=model_config).to(current_config['device'])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=current_config['learning_rates'][0])
        return 0, -1, {}, model, tokenizer, optimizer
    print(f"Length of current stats file: {len(ckpt_info['info'])}\n")

    epoch = ckpt_info["info"][-1]["epoch"]
    idx = ckpt_info["info"][-1]["idx"]
    print("Loading model:", join(model_dir, f"{name}-{epoch}"), f"from epoch {epoch} at index {idx}")
    tokenizer = RobertaTokenizer.from_pretrained(join(model_dir, f"{name}-{epoch}"))
    model = RobertaForSequenceClassification.from_pretrained(join(model_dir, f"{name}-{epoch}")).to(current_config['device'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=current_config['learning_rates'][0])
    return epoch, idx, ckpt_info, model, tokenizer, optimizer
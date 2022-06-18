
# import opendatasets as od
# import transformers
import pandas as pd
import torch
from torch.utils.data import DataLoader

import random 
import os
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from transformers import RobertaTokenizer
from tqdm import tqdm
import time

from src.data import *
from src.model import *
from src.lookup import *
from src.summarize import *
from src.checkpoint_util import *
from src.configs import config_FakeNews

import warnings
warnings.filterwarnings("ignore")


all_methods = ["full","three1one2","three1","one1","none"]
FAKE_FILENAME = 'Fake.csv'
TRUE_FILENAME = 'True.csv'
config = config_FakeNews


seed = config.seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="|".join(all_methods))
    opts = parser.parse_args()
    
    if opts.method not in all_methods:
        raise Exception(f"Wrong method: Method must be in {all_methods}, got {opts.method}")
    return opts

def load_data(dirpath="data/fake-and-real-news-dataset/"):
    fake_df = pd.read_csv(os.path.join(dirpath, FAKE_FILENAME))
    true_df = pd.read_csv(os.path.join(dirpath, TRUE_FILENAME))

    fake_df['news'] = fake_df['text']
    fake_df['news'] = fake_df['title'] + '\n' + fake_df['news']
    fake_df['label'] = 0

    true_df['news'] = true_df['text']
    true_df['news'] = true_df['title'] + '\n' + true_df['news']
    true_df['label'] = 1

    all_data = pd.concat([true_df, fake_df])[['news', 'label']].sample(frac=1).reset_index(drop=True)
    train_val_df, test_df = train_test_split(all_data, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42)

    train_dataset = NewsClsDataset(train_df, tokenizer, train_config['maxlen'])
    val_dataset =  NewsClsDataset(val_df, tokenizer, train_config['maxlen'])
    test_dataset = NewsClsDataset(test_df, tokenizer, train_config['maxlen'])

    # TRAIN DATASET AND VALID DATASET
    train_params = {'batch_size': train_config['train_batch_size'],
                    'shuffle': True,
                    'num_workers': 2,
                    'pin_memory':True
                    }

    test_params = {'batch_size': train_config['valid_batch_size'],
                    'shuffle': False,
                    'num_workers': 2,
                    'pin_memory':True
                    }
    train_data = DataLoader(train_dataset, **train_params)
    val_data = DataLoader(val_dataset, **train_params)
    test_data = DataLoader(test_dataset, **test_params)

    print("Distribution of train_val:", sum(train_val_df.label)/len(train_val_df))
    print("Distribution of test:", sum(test_df.label)/len(test_df))
    return train_data, val_data, test_data


def process_summarized_tokens(summarized_tokens_list, method, lt):
    if method == "none":
        return None
    
    summarized_tokens = summarized_tokens_list[0].detach()
    if method == "one1":
        return summarized_tokens
    
    if method == "full":
        max_order = None
    elif method == "three1one2":
        max_order = 2
    elif method == "three1": 
        max_order = 1
    else:
        raise Exception(f"Unknown method {method}, expect in {all_methods}")
    
    # Summary hierarchy
    lt.add(summarized_tokens, model, optimizer, labels, i, length, max_order=max_order)

    # This part should be in the LookupTable as well
    # nth
    max_order_available = max(lt.summarized_tokens_hierarchy.keys())
    max_order_available = max_order_available if len(lt.summarized_tokens_hierarchy[max_order_available]["summary_tokens"]) < 2 \
                                                else max_order_available-1
    if method != "full" or max_order_available <= 2:
        l = []
    else:
        l = [lt.summarized_tokens_hierarchy[
            max_order_available
        ]["summary_tokens"][0]]

    # 1 2nd
    if method == "three1" or \
            max_order_available < 2 or \
            len(lt.summarized_tokens_hierarchy[2]["summary_tokens"]) < 2: #
        pass
    elif lt.summarized_tokens_hierarchy[2]["summary_tokens"][-1].shape[0] == 0:
        l.append(lt.summarized_tokens_hierarchy[2]["summary_tokens"][-2])
    else:
        l.append(lt.summarized_tokens_hierarchy[2]["summary_tokens"][-1])

    # 3 1st
    if lt.summarized_tokens_hierarchy[1]["summary_tokens"][-1].shape[0] == 0:
        l.extend(lt.summarized_tokens_hierarchy[1]["summary_tokens"][-4:-1])
    else:
        l.extend(lt.summarized_tokens_hierarchy[1]["summary_tokens"][-3:])

    summarized_tokens = torch.concat(l)
    return summarized_tokens


if __name__ == "__main__":
    opts = parse_args()
    
    train_config = config.train_config

    # Hyperparameters for summarization
    N = config.N
    n = config.n # Input tokens size
    m = config.m # Number of summaries in each window
    k = config.k # Number of lookups in each window
    p = config.p
    assert m > k, "Should be m > k"
    assert n < N, "Should be n < N"
    assert k < (N-1)/n - 1, f"Should be k < (N-1)/n - 1, but {k} < {(N-1)/n - 1}"
    assert 0 <= p and p <= ((N-1)-n*(1+k))/(N*(m-k)), f"0 <= {p} <= {((N-1)-n*(1+k))/(N*(m-k))} is not satisfied"

    # Load tokenizer
    use_finetuned = False # default is False
    model_config = config.model_config
    if use_finetuned:
        tokenizer = RobertaTokenizer.from_pretrained("models")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(train_config['model'])
        
    model_name = f"run_{opts.method}"
    model_dir = "models"
    
    train_data, val_data, test_data = load_data()
    

    # We warmed up with 128 indices before inserting tokens into embedding layer
    last_epoch, last_idx, ckpt_info, model, tokenizer, optimizer = load_checkpoint(model_dir, model_name, train_config, model_config)
    # print(ckpt_info)
    # raise
    if last_epoch == 0:
        last_epoch = 1
        nb_tr_steps = 0
        nb_tr_examples = 0
        tr_losses = 0
        tr_accuracies = 0
    else:
        nb_tr_steps = ckpt_info["info"][-1]["Num Tr Steps"]
        nb_tr_examples = ckpt_info["info"][-1]["Num Tr Examples"]
        tr_losses = ckpt_info["info"][-1]["Training Loss"] * nb_tr_steps
        tr_accuracies = ckpt_info["info"][-1]["Training Acc"] * nb_tr_examples
        for i in range(len(ckpt_info["info"])):
            epoch = ckpt_info["info"][i]["epoch"]
            idx = ckpt_info["info"][i]["idx"]
            train_loss, val_loss = ckpt_info["info"][i]["Training Loss"], ckpt_info["info"][i]["Valid. Loss"]
            train_acc, val_acc = ckpt_info["info"][i]["Training Acc"], ckpt_info["info"][i]["Valid. Acc"]
            train_time, val_time = ckpt_info["info"][i]["Training Time"], ckpt_info["info"][i]["Valid. Time"]
            other_info = ckpt_info["info"][i]["Other"]
            print(f"Epoch: {epoch}, index: {idx}, Learning rate = {train_config['learning_rates'][epoch]}")
            print(f"Training loss epoch: {train_loss}")
            print(f"Training accuracy epoch: {train_acc}, elapsed: {train_time}")
            print(f"Val loss epoch: {val_loss}")
            print(f"Val accuracy epoch: {val_acc}, elapsed: {val_time}")
            print(f"Other: {other_info}")
            print()

    for epoch in range(last_epoch, train_config['epochs']+1):
        print(f"Epoch: {epoch}, Learning rate = {train_config['learning_rates'][epoch]}")
        for g in optimizer.param_groups: 
            g['lr'] = train_config['learning_rates'][epoch]
        lr = optimizer.param_groups[0]['lr']

        # Training
        train_time = 0

        model.train()
        for idx, batch in enumerate(tqdm(train_data)):
            if idx <= last_idx:            
                continue

            start_time = time.time()
            # Use 30 max last tokens, split into 3, feed first ten into next 10 and so on.
            # Then train using the last 10.
            summarized_tokens = None
            length = batch["input_ids"].shape[1]
            lt = LookupTable(train_config, N, n, m, k, p)
            for i in range(length // n): # Minus the BOS token

                # Need to use batch size of 1 for now
                assert batch["input_ids"].shape[0] == 1, "Batch size must be 1 for now"

                ids = torch.zeros(1, 512)
                mask = torch.zeros(1, 512)
                end = min( n*(i+1), length )
                length_input_tokens = end - n*i

                ids[0,0] = 0 # BOS token
                ids[0,1:length_input_tokens+1] = batch['input_ids'][0, n*i : end].clone()
                ids[0,length_input_tokens+1:] = 1 # Padding
                mask[0,:length_input_tokens+1] = 1
                mask[0,length_input_tokens+1:] = 0 # Masks the padding
                
                ids = ids.to(train_config['device'], dtype = torch.long) #[C, seq_length]
                mask = mask.to(train_config['device'], dtype = torch.long)
                labels = batch['labels'].to(train_config['device'], dtype = torch.long)

                (loss, tr_logits), summarized_tokens_list = model(input_ids=ids, 
                                        attention_mask=mask, 
                                        labels=labels,
                                        return_dict=False,
                                        summarized_tokens=summarized_tokens)
                
                # Track losses 
                tr_losses += loss.item()
                nb_tr_steps += 1
                nb_tr_examples += labels.size(0)

                # compute training accuracy
                flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
                flattened_predictions = torch.argmax(tr_logits, axis=1)

                tr_accuracies += sum(flattened_predictions==flattened_targets)

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=train_config['max_grad_norm']
                )

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Our method
                summarized_tokens = process_summarized_tokens( summarized_tokens_list, opts.method, lt )
                

            train_time += time.time() - start_time

            if idx % 10 == 0: #TODO: Magic number
                train_loss = tr_losses / nb_tr_steps
                train_acc = tr_accuracies / nb_tr_examples
    #             print(tr_losses)
                print(f"Epoch_loss:", train_loss)
                print("Acc:", train_acc)
                train_time /= 5

                save_checkpoint(model, tokenizer,
                                model_dir, model_name, train_config, 
                                epoch, idx,
                                train_loss, 0, train_acc, 0, train_time, 0,
                                nb_tr_steps, nb_tr_examples,
                                {})
                train_time = 0

        break #One epoch is enough
        
    # Evaluation 
    eval_time = 0

    nb_ev_steps = 0
    nb_ev_examples = 0
    ev_losses = 0
    ev_accuracies = 0

    model.eval()
    for idx, batch in enumerate(tqdm(val_data)):
        start_time = time.time()
        # Use 30 max last tokens, split into 3, feed first ten into next 10 and so on.
        # Then train using the last 10.
        summarized_tokens = None
        length = batch["input_ids"].shape[1]
        lt = LookupTable(train_config, N, n, m, k, p)
        for i in range(length // n): # Minus the BOS token

            # Need to use batch size of 1 for now
            assert batch["input_ids"].shape[0] == 1, "Batch size must be 1 for now"

            ids = torch.zeros(1, 512)
            mask = torch.zeros(1, 512)
            end = min( n*(i+1), length )
            length_input_tokens = end - n*i

            ids[0,0] = 0 # BOS token
            ids[0,1:length_input_tokens+1] = batch['input_ids'][0, n*i : end].clone()
            ids[0,length_input_tokens+1:] = 1 # Padding
            mask[0,:length_input_tokens+1] = 1
            mask[0,length_input_tokens+1:] = 0 # Masks the padding

            ids = ids.to(train_config['device'], dtype = torch.long) #[C, seq_length]
            mask = mask.to(train_config['device'], dtype = torch.long)
            labels = batch['labels'].to(train_config['device'], dtype = torch.long)

            (loss, ev_logits), summarized_tokens_list = model(input_ids=ids, 
                                    attention_mask=mask, 
                                    labels=labels,
                                    return_dict=False,
                                    summarized_tokens=summarized_tokens)
            summarized_tokens = summarized_tokens_list[0].detach()
    #             print(summarized_tokens.shape)

            # Track three types of losses separately
            ev_losses += loss.item()
            nb_ev_steps += 1
            nb_ev_examples += labels.size(0)


            # compute training accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            flattened_predictions = torch.argmax(ev_logits, axis=1)

            ev_accuracies += sum(flattened_predictions==flattened_targets)

            # Summary hierarchy
            lt.add(summarized_tokens, model, optimizer, labels, i, length, is_training=False)

            # This part should be in the LookupTable as well
            # nth
            max_order = max(lt.summarized_tokens_hierarchy.keys())
            max_order = max_order if len(lt.summarized_tokens_hierarchy[max_order]["summary_tokens"]) < 2 else max_order-1
            if max_order <= 2:
                l = []
            else:
                l = [lt.summarized_tokens_hierarchy[
                    max_order
                ]["summary_tokens"][0]]
            # l = []

            # 1 2nd
            if max_order < 2 or len(lt.summarized_tokens_hierarchy[2]["summary_tokens"]) < 2:
                pass
            elif lt.summarized_tokens_hierarchy[2]["summary_tokens"][-1].shape[0] == 0:
                l.append(lt.summarized_tokens_hierarchy[2]["summary_tokens"][-2])
            else:
                l.append(lt.summarized_tokens_hierarchy[2]["summary_tokens"][-1])

            # 3 1st
            if lt.summarized_tokens_hierarchy[1]["summary_tokens"][-1].shape[0] == 0:
                l.extend(lt.summarized_tokens_hierarchy[1]["summary_tokens"][-4:-1])
            else:
                l.extend(lt.summarized_tokens_hierarchy[1]["summary_tokens"][-3:])

            summarized_tokens = torch.concat(l)

        eval_time += time.time() - start_time

        if idx % 10 == 0: #TODO: Magic number
            eval_loss = ev_losses / nb_ev_steps
            eval_acc = ev_accuracies / nb_ev_examples
    #             print(tr_losses)
            print(f"Epoch_loss:", eval_loss)
            print("Acc:", eval_acc)
            eval_time /= 5
            eval_time = 0
            
    eval_loss = ev_losses / nb_ev_steps
    eval_acc = ev_accuracies / nb_ev_examples
    print(f"Epoch_loss:", eval_loss)
    print("Acc:", eval_acc)
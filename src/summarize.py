"""
Summarization modules

Author: 
"""
import torch
import torch.nn

def summarize(t, model, labels, optimizer):
    if t.shape[0] == 0:
        return None
    
    ids = torch.zeros(1, 512)
    mask = torch.zeros(1, 512)
    end = min( n*(i+1), length )
    length_input_tokens = end - n*i

    ids[0,0] = 0 # BOS token
    ids[0,1:] = 1
    mask[0,:1] = 1
    mask[0,1:] = 0 # Masks the padding
    ids = ids.to(config['device'], dtype = torch.long) #[C, seq_length]
    mask = mask.to(config['device'], dtype = torch.long)
    (loss, tr_logits), summarized_tokens_list = model(input_ids=ids, # BOS 
                                    attention_mask=mask, 
                                    labels=labels,
                                    return_dict=False,
                                    summarized_tokens=t)
    summarized_tokens = summarized_tokens_list[0].detach()

    # compute training accuracy
    flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
    flattened_predictions = torch.argmax(tr_logits, axis=1)

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(
        parameters=model.parameters(), max_norm=config['max_grad_norm']
    )

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return summarized_tokens
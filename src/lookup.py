"""
Lookup modules

Author: 
"""

import torch
import torch.nn


class LookupTable:
    def __init__(self, 
                 train_config,
                 N = 45, # Context window size
                 n = 15, # Input tokens size
                 m = 5, # Number of summaries in each window
                 k = 1, # Number of lookups in each window
                 p = 1/13):
        self.N = N
        self.n = n
        self.m = m
        self.k = k
        self.p = p
        self.train_config = train_config
        
        self.summarized_tokens_hierarchy = {}


    def add(self, summarized_tokens, model, optimizer, labels, i, length, 
            order=1, 
            add_new=True,
            max_order=None,
            is_training=True,
           ):
        if order not in self.summarized_tokens_hierarchy:
            self.summarized_tokens_hierarchy[order] = {
                "summary_tokens": [torch.ones(0)],
                "sum_from": 0
            }

        self.summarized_tokens_hierarchy[order]["summary_tokens"][-1] = summarized_tokens
        if add_new:
            self.summarized_tokens_hierarchy[order]["summary_tokens"].append(torch.ones(0))
        else:
            pass

        sum_from = self.summarized_tokens_hierarchy[order]["sum_from"]
    #     print(f"Length at order {order}: {len(summarized_tokens_hierarchy[order]['summary_tokens'])}")
    #     print(f"Total length at order {order}: {sum([x.shape[0] for x in summarized_tokens_hierarchy[order]['summary_tokens'][sum_from:]])}")

        if max_order != None and order >= max_order:
            pass
        elif sum([x.shape[0] for x in self.summarized_tokens_hierarchy[order]["summary_tokens"][sum_from:]]) > self.n:
            # Create higher order summaries
    #         print(f"Real Summarizing at {i}, order={order}, sum from {sum_from}")

            input_tokens = torch.concat([x for x in self.summarized_tokens_hierarchy[order]["summary_tokens"][sum_from:-1]], axis=0)
            summarized_tokens_next = self.summarize(input_tokens, model, labels, optimizer, i, length, is_training=is_training,)
            self.add(summarized_tokens_next, model, optimizer, labels, i, length, order=order+1, add_new=True, max_order=max_order, is_training=is_training,)

            # self.summarized_tokens_hierarchy[order]["summary_tokens"] = self.summarized_tokens_hierarchy[order]["summary_tokens"][sum_from:] # don't include the far summaries
            sum_from = len(self.summarized_tokens_hierarchy[order]["summary_tokens"]) - 1
        elif sum_from == 0:
            if len(self.summarized_tokens_hierarchy[order]["summary_tokens"]) <= 1:
                pass
            elif len(self.summarized_tokens_hierarchy[order]["summary_tokens"]) == 2 and \
                    self.summarized_tokens_hierarchy[order]["summary_tokens"][-1].shape[0] == 0:
                pass
            else:
    #             print(f"Summarizing at {i}, order={order}, sum from {sum_from}")
                input_tokens = torch.concat([x for x in self.summarized_tokens_hierarchy[order]["summary_tokens"][sum_from:-1]], axis=0)
                summarized_tokens_next = self.summarize(input_tokens, model, labels, optimizer, i, length, is_training=is_training,)
                if summarized_tokens_next == None:
                    pass
                else:
                    self.add(summarized_tokens_next, model, optimizer, labels, i, length, order=order+1, add_new=False, max_order=max_order, is_training=is_training,)
        else:
            self.summarized_tokens_hierarchy[order]["summary_tokens"].pop(0)
            sum_from -= 1

            if len(self.summarized_tokens_hierarchy[order]["summary_tokens"]) <= 1:
                pass
            elif len(self.summarized_tokens_hierarchy[order]["summary_tokens"]) == 2 and \
                    self.summarized_tokens_hierarchy[order]["summary_tokens"][-1].shape[0] == 0:
                pass
            else:
    #             print(f"Summarizing at {i}, order={order}, sum from {sum_from}")
                try:
                    input_tokens = torch.concat([x for x in self.summarized_tokens_hierarchy[order]["summary_tokens"][sum_from:-1]], axis=0)
                    summarized_tokens_next = self.summarize(input_tokens, model, labels, optimizer, i, length, is_training=is_training,)
                    if summarized_tokens_next == None:
                        pass
                    else:
                        self.add(summarized_tokens_next, model, optimizer, labels, i, length, order=order+1, add_new=False, max_order=max_order, is_training=is_training,)
                except Exception as e:
                    if "NotImplementedError" in repr(e):
                        pass
                    else:
                        print(repr(e))
                        raise

        self.summarized_tokens_hierarchy[order]["sum_from"] = sum_from
        
    def summarize(self, t, model, labels, optimizer,  i, length, is_training=True,):
        if t.shape[0] == 0:
            return None

        ids = torch.zeros(1, 512)
        mask = torch.zeros(1, 512)
        end = min( self.n*(i+1), length )
        length_input_tokens = end - self.n*i

        ids[0,0] = 0 # BOS token
        ids[0,1:] = 1
        mask[0,:1] = 1
        mask[0,1:] = 0 # Masks the padding
        ids = ids.to(self.train_config['device'], dtype = torch.long) #[C, seq_length]
        mask = mask.to(self.train_config['device'], dtype = torch.long)
        (loss, tr_logits), summarized_tokens_list = model(input_ids=ids, # BOS 
                                        attention_mask=mask, 
                                        labels=labels,
                                        return_dict=False,
                                        summarized_tokens=t)
        summarized_tokens = summarized_tokens_list[0].detach()

        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        flattened_predictions = torch.argmax(tr_logits, axis=1)

        
        if is_training:
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=self.train_config['max_grad_norm']
            )

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return summarized_tokens
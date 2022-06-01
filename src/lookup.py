"""
Lookup modules

Author: 
"""


def add_into_dictionary(summarized_tokens, summarized_tokens_hierarchy, model, labels, order=1, add_new=True):
    if order not in summarized_tokens_hierarchy:
        summarized_tokens_hierarchy[order] = {
            "summary_tokens": [torch.ones(0)],
            "sum_from": 0
        }
        
    summarized_tokens_hierarchy[order]["summary_tokens"][-1] = summarized_tokens
    if add_new:
        summarized_tokens_hierarchy[order]["summary_tokens"].append(torch.ones(0))
    else:
        pass
        
    sum_from = summarized_tokens_hierarchy[order]["sum_from"]
#     print(f"Length at order {order}: {len(summarized_tokens_hierarchy[order]['summary_tokens'])}")
#     print(f"Total length at order {order}: {sum([x.shape[0] for x in summarized_tokens_hierarchy[order]['summary_tokens'][sum_from:]])}")
    
    # Higher order summaries
    if sum([x.shape[0] for x in summarized_tokens_hierarchy[order]["summary_tokens"][sum_from:]]) > n:
#         print(f"Real Summarizing at {i}, order={order}, sum from {sum_from}")

        input_tokens = torch.concat([x for x in summarized_tokens_hierarchy[order]["summary_tokens"][sum_from:-1]], axis=0)
        summarized_tokens_next = summarize(input_tokens, model, labels)
        add_into_dictionary(summarized_tokens_next, summarized_tokens_hierarchy, model, labels, order=order+1, add_new=True)

        summarized_tokens_hierarchy[order]["summary_tokens"] = summarized_tokens_hierarchy[order]["summary_tokens"][sum_from:] # don't include the far summaries
        sum_from = len(summarized_tokens_hierarchy[order]["summary_tokens"]) - 1
    elif sum_from == 0:
        if len(summarized_tokens_hierarchy[order]["summary_tokens"]) <= 1:
            pass
        elif len(summarized_tokens_hierarchy[order]["summary_tokens"]) == 2 and \
                summarized_tokens_hierarchy[order]["summary_tokens"][-1].shape[0] == 0:
            pass
        else:
#             print(f"Summarizing at {i}, order={order}, sum from {sum_from}")
            input_tokens = torch.concat([x for x in summarized_tokens_hierarchy[order]["summary_tokens"][sum_from:-1]], axis=0)
            summarized_tokens_next = summarize(input_tokens, model, labels)
            if summarized_tokens_next == None:
                pass
            else:
                add_into_dictionary(summarized_tokens_next, summarized_tokens_hierarchy, model, labels, order=order+1, add_new=False)
    else:
        summarized_tokens_hierarchy[order]["summary_tokens"].pop(0)
        sum_from -= 1
        
        if len(summarized_tokens_hierarchy[order]["summary_tokens"]) <= 1:
            pass
        elif len(summarized_tokens_hierarchy[order]["summary_tokens"]) == 2 and \
                summarized_tokens_hierarchy[order]["summary_tokens"][-1].shape[0] == 0:
            pass
        else:
#             print(f"Summarizing at {i}, order={order}, sum from {sum_from}")
            try:
                input_tokens = torch.concat([x for x in summarized_tokens_hierarchy[order]["summary_tokens"][sum_from:-1]], axis=0)
                summarized_tokens_next = summarize(input_tokens, model, labels)
                if summarized_tokens_next == None:
                    pass
                else:
                    add_into_dictionary(summarized_tokens_next, summarized_tokens_hierarchy, model, labels, order=order+1, add_new=False)
            except Exception as e:
                if "NotImplementedError" in repr(e):
                    pass
                else:
                    print(repr(e))
                    raise

    summarized_tokens_hierarchy[order]["sum_from"] = sum_from
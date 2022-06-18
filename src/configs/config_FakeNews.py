import torch
from src.model import RobertaConfig


# For reproducibility
seed = 1293182
train_config  =  {'model':"roberta-base",
            'maxlen' :512,
            'train_batch_size':1,
            'valid_batch_size':1,
            'epochs':1,
#             'learning_rates':[0.25e-4, 0.25e-4, 0.25e-4, 0.25e-4, 0.25e-5], #Custom lr proven to work well in my previous tasks
            'learning_rates':[0.1e-5, 0.1e-5, 0.25e-4, 0.25e-4, 0.25e-5],
            'max_grad_norm':10,
            'device':'cuda' if torch.cuda.is_available() else 'cpu'}

# Hyperparameters for summarization
N = 45
n = 15 # Input tokens size
m = 5 # Number of summaries in each window
k = 1 # Number of lookups in each window
p = 1/13
assert m > k, "Should be m > k"
assert n < N, "Should be n < N"
assert k < (N-1)/n - 1, f"Should be k < (N-1)/n - 1, but {k} < {(N-1)/n - 1}"
assert 0 <= p and p <= ((N-1)-n*(1+k))/(N*(m-k)), f"0 <= {p} <= {((N-1)-n*(1+k))/(N*(m-k))} is not satisfied"

# Model configs
use_finetuned = False # default is False
model_config = RobertaConfig(
    _name_or_path="roberta-base",
    architectures=["RobertaForMaskedLM"],
    bos_token_id= 0,
    classifier_dropout=None,
    eos_token_id=2,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=768,
    initializer_range=0.02,
    intermediate_size=3072,
    layer_norm_eps=1e-05,
    max_position_embeddings=514,
    model_type="roberta",
    num_attention_heads=12,
    num_hidden_layers=12,
    pad_token_id=1,
    position_embedding_type="absolute",
    transformers_version="4.15.0",
    type_vocab_size=1,
    use_cache=True,
    vocab_size=50265,
    summarize_layers = [11],
    shrink_percentage = p,
    num_labels=2
)
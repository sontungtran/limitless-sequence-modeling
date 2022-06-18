import torch
from torch.utils.data import Dataset

class NewsClsDataset(Dataset):
    def __init__(self, df, tokenizer, maxlen=512):
        self.data = df
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        
    def __getitem__(self, index):
        text = self.data.news.values[index]
        labels = self.data.label.values[index]
        
        encoded = self.tokenizer(text.split(),
                                is_split_into_words=True,)
#                                 padding='max_length',
#                                 truncation=True,
#                                 max_length=self.maxlen)
        
        input_ids = encoded['input_ids'][1:]
        masks = encoded['attention_mask'][1:]        
            
        item = {'input_ids': torch.tensor(input_ids),
                'attention_masks': torch.tensor(masks),
                'labels': torch.tensor(labels)}

        return item
    
    def __len__(self):
        return len(self.data)
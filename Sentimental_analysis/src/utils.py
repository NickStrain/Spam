import pandas as pd 
import numpy as np
import transformers
from transformers import BertModel, BertTokenizer
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


df = pd.read_csv(r"./IMDB Dataset.csv")
df_train, df_test = train_test_split(df,test_size=0.2,random_state=42)
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
MAX_LEN = 160

# df_train.set_index("index",inplace=True)
# df_test.set_index("index",inplace=True)

# def tokenizer_BERT(tokenizer,text):
#     '''
#     This func used to token the text
#     '''
    
#     encoding = tokenizer.encode_plus(
#     text,
#     max_length=32,
#     add_special_tokens=True, # Add '[CLS]' and '[SEP]'
#     return_token_type_ids=False,
#     pad_to_max_length=True,
#     return_attention_mask=True,
#     return_tensors='pt',  # Return PyTorch tensors
#     )

#     return encoding


class IMDB(Dataset):
    def __init__(self,df,tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        self.rev = self.df['review']
        self.lab = self.df['sentiment']
        ta = 0
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        if self.lab[index] =='negative':
            ta=1
        else:
            ta =0
        encoding = self.tokenizer.encode_plus(
        self.rev[index],
        add_special_tokens=True,
        max_length=150,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
            
        return {
            'review_text': self.rev[index],
            'input_ids': encoding['input_ids'].flatten(),
            'attentioin_mask':encoding['attention_mask'].flatten(),
            'targets':torch.tensor(ta, dtype=torch.long)
        }



if __name__ == '__main__':
    pass
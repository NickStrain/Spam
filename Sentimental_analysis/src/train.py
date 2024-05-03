import pandas as pd 
import numpy as np
import transformers
from transformers import BertModel, BertTokenizer,AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, random_split
from utils import IMDB
from tqdm import tqdm 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from model import SentimentClassifier

df = pd.read_csv(r"./IMDB Dataset.csv")
df_train, df_test = train_test_split(df,test_size=0.2,random_state=42)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertModel.from_pretrained('bert-base-cased')
MAX_LEN = 160

df_train.reset_index(drop=True,inplace=True)
df_test.reset_index(drop=True,inplace=True)

data_train = IMDB(df_train,tokenizer)
data_test = IMDB(df_test,tokenizer)

train_loader = DataLoader(data_train,batch_size=16)
test_loader = DataLoader(data_test,batch_size=16)

'''
Model :)
'''
model = SentimentClassifier(2)
EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)


def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):
  model = model.train()
  losses = []
  correct_predictions = 0
  for d in tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attentioin_mask"].to(device)
    targets = d["targets"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double() / n_examples, np.mean(losses)

for e in range(5):
    train_acc, train_loss = train_epoch(
    model,
    train_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
  )
    print(f"train loss{train_loss}")
    
if __name__ == '__main__':
    pass
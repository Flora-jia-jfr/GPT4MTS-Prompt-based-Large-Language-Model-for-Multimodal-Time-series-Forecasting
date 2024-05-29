import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from pathlib import Path
import pickle
from transformers import BertTokenizer, BertModel
import torch
import re

warnings.filterwarnings('ignore')

class Dataset_GDELT(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_dir='ts_data/', data_filename='CH00.csv',
                 event_root_code = 1, target='NumMentions', 
                 scale=True, freq='d', percent=100, 
                 max_len=-1, train_all=False, channel_independent=False,
                 summary=False):
        if size == None:
            raise Exception("Please indicate the seq_len, label_len, pred_len")
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_dir = data_dir
        self.data_filename = data_filename
        self.event_root_code = event_root_code
        self.summary = summary
        self.channel_independent = channel_independent
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

        if self.tot_len < 0:
            return None

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_dir, self.data_filename))
        
        print(f"Filter for {self.data_filename, self.event_root_code} ...")
        df_raw = df_raw[df_raw['EventRootCode'] == self.event_root_code]

        border1s = [0, int(0.7 * len(df_raw)) - self.seq_len, int(0.9 * len(df_raw)) - self.seq_len]
        border2s = [int(0.7 * len(df_raw)), int(0.9 * len(df_raw)), len(df_raw)]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        df_stamp = df_raw[['Date']][border1:border2]


        if self.scale:
            train_data = df_raw[border1s[0]:border2s[0]]
            self.scaler.fit(train_data[['NumMentions', 'NumSources', 'NumArticles']].values)
            scaled_columns = self.scaler.transform(df_raw[['NumMentions', 'NumSources', 'NumArticles']].values)
            df_raw[['NumMentions', 'NumSources', 'NumArticles']] = scaled_columns
        
        df_data = df_raw[border1:border2]

        if self.features == 'M' or self.features == 'MS':
            self.data_x = df_data[['NumMentions', 'NumSources', 'NumArticles']].values
            self.data_y = df_data[['NumMentions', 'NumSources', 'NumArticles']].values
        elif self.features == 'S':
            self.data_x = df_data[[self.target]]
            self.data_y = df_data[[self.target]]

        if self.summary:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased").to('cuda')

            texts = df_data['summary'].values.tolist()
            texts = [re.sub(r'[^a-zA-Z0-9 ]+', "", text) if isinstance(text, str) else "Predict the future time step given the past" for text in texts]

            batch_size = 8
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')
                with torch.no_grad():
                    outputs = model(**inputs)
                # [CLS]
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(batch_embeddings)

            if all_embeddings:
                all_embeddings = torch.cat(all_embeddings, dim=0)
                self.data_s = all_embeddings.to('cpu')

        data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

    def __len__(self):
        return max(self.tot_len * self.enc_in, 0)

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        if self.channel_independent:
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            if self.features == 'M' or self.features == 'MS':
                seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            elif self.features == 'S':
                seq_y = self.data_y[r_begin:r_end]
        else:
            seq_x = self.data_x[s_begin:s_end, :]
            seq_y = self.data_y[r_begin:r_end, :]
        
        seq_s = self.data_s[s_begin:s_end, :]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)
        seq_s = torch.tensor(seq_s, dtype=torch.float32)
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_s

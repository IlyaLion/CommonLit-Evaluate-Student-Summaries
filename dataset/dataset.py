import torch
from transformers import AutoTokenizer

import os
import pandas as pd
import numpy as np


class CommonLitDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.full_text_tokens = df['full_text_tokens'].values
        self.labels = df[['content', 'wording']].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tokens = self.full_text_tokens[index]
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return tokens, label
    
class CommonLitDataModule:
    def __init__(self, config):
        self.config = config
        prompts_train_df = pd.read_csv(os.path.join(self.config.directories.data, 'prompts_train.csv'))
        summaries_train_df = pd.read_csv(os.path.join(self.config.directories.data, 'summaries_train.csv'))
        df = prompts_train_df.merge(summaries_train_df, on='prompt_id')
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.backbone_type)
        df['full_text'] = (df['prompt_question'].values + tokenizer.sep_token +
                           df['text'].values)
        
        full_text_tokens = []
        for i in range(len(df)):
            tokens = tokenizer.encode_plus(df.loc[i, 'full_text'], 
                                           max_length=config.tokenizer.max_length,
                                           truncation=True,
                                           add_special_tokens=True,
                                           padding='max_length')
            full_text_tokens.append({'input_ids': torch.tensor(tokens['input_ids'], dtype=torch.long),
                                     'attention_mask': torch.tensor(tokens['attention_mask'], dtype=torch.long)})
        df['full_text_tokens'] = full_text_tokens

        promt_ids = np.sort(df['prompt_id'].unique())

        self.train_df = df[df['prompt_id'] != promt_ids[config.fold]].reset_index(drop=True)
        self.train_df['tokens_len'] = self.train_df['full_text_tokens'].apply(lambda t: t['attention_mask'].sum().item())
        self.train_df = self.train_df.sort_values('tokens_len').reset_index(drop=True)

        self.val_df = df[df['prompt_id'] == promt_ids[config.fold]].reset_index(drop=True)
        self.val_df['tokens_len'] = self.val_df['full_text_tokens'].apply(lambda t: t['attention_mask'].sum().item())
        self.val_df = self.val_df.sort_values('tokens_len').reset_index(drop=True)

    def train_dataloader(self):

        dataset = CommonLitDataset(self.train_df)
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=self.config.data_loaders.train.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=False,
                                           drop_last=True)

    def val_dataloader(self):
        dataset = CommonLitDataset(self.val_df)
        return torch.utils.data.DataLoader(dataset=dataset, 
                                           batch_size=self.config.data_loaders.val.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=False,
                                           drop_last=False)
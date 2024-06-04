import os
import pandas as pd
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.transforms import SentencePieceTokenizer, CLIPTokenizer
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

MERGES_FILE = "http://download.pytorch.org/models/text/clip_merges.bpe"
ENCODER_FILE = "http://download.pytorch.org/models/text/clip_encoder.json"



class AnnoMI_Dataset(Dataset):
    def __init__(self, data_pth='./data/',filename="AnnoMI-full.csv", **kwargs) -> None:
        super().__init__()
        self.filename = filename
        self.og_data = pd.read_csv(os.path.join(data_pth, self.filename))
        self.data = self.filter_data(self.og_data)
        # 116/544, 69/544, 145/544, 214/544
        self.label_dict = {"question": 0, "therapist_input": 1, "reflection": 2, "other": 3}
        # self.tokenizer = CLIPTokenizer(merges_path=MERGES_FILE, encoder_json_path=ENCODER_FILE)
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        # print(self.data.columns, self.data["utterance_text"][4], self.data ["main_therapist_behaviour"][4], self.tokenizer(self.data["utterance_text"][4]))
        
        self.data = self.data.reset_index(drop=True)
        # print(self.data)


    def filter_data(self, pd_data) -> pd.DataFrame:
        """
            Input: takes in a pandas dataframe
            Output: outputs a filtered pandas dataframe based on requirements 
            For eg: removing all client samples and low mi samples from the original dataframe
        """
        # print("before", len(pd_data.loc[pd_data['transcript_id'] == 55]))
        # pd_data = pd_data.drop_duplicates( 
        #     subset = ['utterance_text', 'timestamp','transcript_id'], 
        #     keep = 'last').reset_index(drop = True)
        # print("after deduplication",len(pd_data.loc[pd_data['transcript_id'] == 55]))
        return pd_data.loc[pd_data['interlocutor'] == "therapist"][["utterance_id", "utterance_text", "client utterance", "main_therapist_behaviour"]]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> List:
        client_text = self.data["client utterance"][index]
        # print(client_text)
        # x, y = self.tokenizer((("<client> " + client_text), ("<therapist> " + self.data["utterance_text"][index]))), self.label_dict[self.data["main_therapist_behaviour"][index]]
        x, y = self.tokenizer("<client> " + client_text + "<therapist> " + self.data["utterance_text"][index]), self.label_dict[self.data["main_therapist_behaviour"][index]]
        # x = torch.tensor([int(i) for i in x])

        # print(self.tokenizer.decode(x["input_ids"]))
        y = torch.tensor(y)
        return torch.tensor(x["input_ids"]), torch.tensor(x["attention_mask"]), torch.tensor(x["token_type_ids"]), y

def collate_func(batch):
    inp1, inp2, inp3, labels = zip(*batch)

    # print(inp,labels)
    inp1 = pad_sequence(inp1, padding_value=0, batch_first=True)
    inp2 = pad_sequence(inp2, padding_value=0, batch_first=True)
    inp3 = pad_sequence(inp3, padding_value=0, batch_first=True)
    labels = torch.stack(labels)
    return [inp1,inp2,inp3], labels


# train_dataset = AnnoMI_Dataset(filename="AnnoMI-processed_train.csv")
# test_dataset = AnnoMI_Dataset(filename="AnnoMI-processed_test.csv")
# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_func)
# test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_func)
# print("len of train loader", len(train_dataloader))
# print("len of test loader",len(test_dataloader))
# print(next(iter(train_dataset)))
# train_features, labels = next(iter(train_dataloader))
# print(train_features)
# print(labels.shape)
# # print(f"Feature batch shape: {train_features[0].size()}")
# # print(f"Labels batch shape: {train_features[1].size()}")
# # print(next(iter(train_dataset)))

    


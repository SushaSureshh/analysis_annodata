import os
import pandas as pd
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.transforms import SentencePieceTokenizer, CLIPTokenizer
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

class AnnoMI_Dataset_lstm(Dataset):
    def __init__(self, data_pth='./data/',filename="AnnoMI-full.csv", **kwargs) -> None:
        super().__init__()
        self.filename = filename
        self.og_data = pd.read_csv(os.path.join(data_pth, self.filename))
        self.data = self.filter_data(self.og_data)
        self.label_dict = {"question": 0, "therapist_input": 1, "reflection": 2, "other": 3}
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.data = self.data.reset_index(drop=True)
        self.data_len = self.data.groupby(['transcript_id'], as_index=False).size()


    def filter_data(self, pd_data) -> pd.DataFrame:
        """
            Input: takes in a pandas dataframe
            Output: outputs a filtered pandas dataframe based on requirements 
            For eg: removing all client samples and low mi samples from the original dataframe
        """
        return pd_data.loc[pd_data['interlocutor'] == "therapist"][["transcript_id","utterance_id", "utterance_text", "client utterance", "main_therapist_behaviour"]]
    
    def __len__(self):
        # Length of trainset is dependent on the number of transcript ids
        return len(self.data_len)
    
    def __getitem__(self, index) -> List:
        
        temp_len = self.data_len.loc[(self.data_len.index == index)]
        # print("index", index, temp_len)
        # print(temp["transcript_id"][0])
        temp_data = self.data.loc[(self.data["transcript_id"] == temp_len["transcript_id"][index])]
        temp_data=temp_data.reset_index(drop=True)
        # print("temp data", temp_data)
        final_x = []
        final_x_len = []
        final_y = []
        for i in range(temp_len["size"][index]):
            client_text = temp_data["client utterance"][i]
            x, y = self.tokenizer("<client> " + client_text + "<therapist> " + temp_data["utterance_text"][i]), self.label_dict[temp_data["main_therapist_behaviour"][i]]
            # TODO ensure slicing of the max sequence to prevent outliers from increasing seq length for LSTM
            final_x.append(torch.tensor(x["input_ids"][:300]))
            final_x_len.append(len(x["input_ids"][:300]))
            final_y.append(torch.tensor(y))
        x = pad_sequence(final_x, padding_value=0, batch_first=True)
        x = pack_padded_sequence(x, final_x_len, batch_first=True, enforce_sorted=False)
        x = pad_packed_sequence(x, padding_value=0, batch_first=True, total_length = 300)
        y = torch.stack(final_y)
        return x, y
    
def collate_func(batch):
    '''
    This function is used to collate padding when the sequence length of input is less than max seq length
    '''
    inp1, inp2, inp3, labels = zip(*batch)
    inp1 = pad_sequence(inp1, padding_value=0, batch_first=True)
    inp2 = pad_sequence(inp2, padding_value=0, batch_first=True)
    inp3 = pad_sequence(inp3, padding_value=0, batch_first=True)
    labels = torch.stack(labels)
    return [inp1,inp2,inp3], labels
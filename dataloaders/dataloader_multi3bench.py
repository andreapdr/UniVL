from torch.utils.data import Dataset
import torch
import numpy as np
import json


class Multi3bench_DataLoader(Dataset):

    def __init__(self,
                 tokenizer,
                 datapath="./data/multi3bench/smallChangeOfState.json"):

        self.data = json.load(open(datapath))
        self.idx2data = {i: k for i, k in enumerate(self.data.keys())}
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id):
        caption_text = ["[CLS]"] + self.tokenizer.tokenize(
            str(self.data[video_id]["capt"]).lower()) + ["[SEP]"]
        foil_text = ["[CLS]"] + self.tokenizer.tokenize(
            str(self.data[video_id]["reverse_foil"]).lower()) + ["[SEP]"]
        caption_text = self.tokenizer.convert_tokens_to_ids(caption_text)
        foil_text = self.tokenizer.convert_tokens_to_ids(foil_text)
        input_mask_caption = [1]*len(caption_text)
        input_mask_foil = [1]*len(foil_text)
        token_type_ids_caption = [0]*len(caption_text)
        token_type_ids_foil = [0]*len(foil_text)
        # casting to tensors >>>
        caption_text = torch.tensor(caption_text)
        foil_text = torch.tensor(foil_text)
        input_mask_caption = torch.tensor(input_mask_caption)
        input_mask_foil = torch.tensor(input_mask_foil)
        token_type_ids_caption = torch.tensor(token_type_ids_caption)
        token_type_ids_foil =  torch.tensor(token_type_ids_foil)
        # <<< casting to tensors
        return caption_text, foil_text, input_mask_caption, input_mask_foil, token_type_ids_caption, token_type_ids_foil

    def _get_video(self, feature_idx):
        feature_path = f"./data/multi3bench/{feature_idx}.npy"
        video_features = np.load(feature_path)
        video_mask = torch.ones(size=(1, video_features.shape[0]))
        return video_features, video_mask

    def __getitem__(self, feature_idx):
        video_id = self.idx2data[feature_idx]
        caption, foil, input_mask_caption, input_mask_foil, token_type_caption, token_type_foil = self._get_text(
            video_id)
        video_features, video_mask = self._get_video(video_id)

        return caption, foil, input_mask_caption, input_mask_foil,\
            token_type_caption, token_type_foil, video_features,\
            video_mask

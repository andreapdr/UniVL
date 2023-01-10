from torch.utils.data import Dataset
import torch
import numpy as np
import json
from os.path import expanduser


class Multi3bench_DataLoader(Dataset):

    def __init__(self,
                 tokenizer,
                 tasks = ["action_foil"],
                 datapath="./data/multi3bench/changeOfState.json",
                 feature_dir="~/VideoFeatureExtractor/processed",
                 instrument="change-state"):

        self.data = json.load(open(datapath))
        self.tasks = ["caption"] + tasks
        self.idx2data = {i: k for i, k in enumerate(self.data.keys())}
        self.tokenizer = tokenizer
        self.feature_dir = self._set_feature_dir(feature_dir)
        self.instrument = instrument

    def _set_feature_dir(self, feature_dir):
        if "~" in feature_dir:
            _feature_dir = self.feature_dir = expanduser(
                "~") + "/VideoFeatureExtractor/processed"
        else:
            _feature_dir = feature_dir
        return _feature_dir

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, task):
        if self.instrument == "change-state":
            if task == "caption":
                text = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + self.tokenizer.tokenize(str(self.data[video_id]["caption"]).lower()) + ["[SEP]"])
            else:
                text = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + self.tokenizer.tokenize(str(self.data[video_id]["foils"][task]).lower()) + ["[SEP]"])
        else:
            raise NotImplementedError
        mask = [1]*len(text)
        token_type_ids = [0]*len(text)
        return torch.tensor(text), torch.tensor(mask), torch.tensor(token_type_ids)

    def _get_video(self, feature_idx):
        feature_path = f"{self.feature_dir}/{feature_idx}.npy"
        try:
            video_features = np.load(feature_path)
            video_mask = torch.ones(size=(1, video_features.shape[0]))
        except:
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        return video_features, video_mask

    def __getitem__(self, feature_idx):
        video_id = self.data[str(feature_idx)]["video_file"].split(".")[0]
        texts = {k: self._get_text(str(feature_idx), k) for k in self.tasks}
        video_features, video_mask = self._get_video(video_id)
        return (texts, video_features, video_mask), video_id
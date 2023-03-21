import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)

import sys

sys.path.append(parent_dir)

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import json
from os.path import expanduser
from VideoFeatureExtractor.preprocessing import Preprocessing


class VLBenchDataset(Dataset):
    def __init__(
        self,
        datapath,
        tokenizer,
        video_feature_extractor,
        videodir="~/datasets/vl-bench/videos",
        instrument="change-of-state",
        task="action",
        device="cpu",
        centercrop=True,
    ):
        self._data = json.load(open(datapath))
        self.data = list(self._data.values())
        self.task = task
        self.instrument = instrument
        self.tokenizer = tokenizer
        self.video_feat_extractor = video_feature_extractor
        self.video_preprocessor = Preprocessing(
            type="s3dg", FRAMERATE_DICT={"s3dg": 16}
        )
        self.videodir = os.path.expanduser(videodir)
        self.device = device
        self.centercrop = centercrop
        print(f"- evaluating instrument: {self.instrument} on setting: {self.task}")

    def __len__(self):
        return len(self.data)

    def _get_text(self, idx):
        _capt = self.data[idx]["foils"][self.task][0]
        _foil = self.data[idx]["foils"][self.task][1]
        capt = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + self.tokenizer.tokenize(_capt.lower()) + ["[SEP]"]
        )
        foil = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]"] + self.tokenizer.tokenize(_foil.lower()) + ["[SEP]"]
        )
        mask_capt = [1] * len(capt)
        mask_foil = [1] * len(foil)
        token_type_ids_capt = [0] * len(capt)
        token_type_ids_foil = [0] * len(foil)
        return {
            "capt": torch.tensor(capt),
            "capt_mask": torch.tensor(mask_capt),
            "capt_token_type": torch.tensor(token_type_ids_capt),
            "foil": torch.tensor(foil),
            "foil_mask": torch.tensor(mask_foil),
            "foil_token_type": torch.tensor(token_type_ids_foil),
        }

    def _get_video(self, video_id):
        video_feat = self._extract_features(video_id)
        video_mask = torch.ones(size=(1, video_feat.shape[0]))
        return {"video_feat": video_feat, "video_mask": video_mask}

    def __getitem__(self, idx):
        text = self._get_text(idx)
        video = self._get_video(idx)
        return video, text

    def _extract_features(self, idx):
        video_path = os.path.join(self.videodir, self.data[idx]["video-id"]) + ".mp4"
        video = torchvision.io.read_video(
            video_path,
            pts_unit="sec",
            start_pts=self.data[idx]["timestamp"][0],
            end_pts=self.data[idx]["timestamp"][1],
        )[0].permute(0, 3, 1, 2)

        if self.centercrop:
            video = torchvision.transforms.CenterCrop((224, 224))(video)

        with torch.no_grad():
            self.video_feat_extractor.eval()
            video = video.squeeze()
            video = self.video_preprocessor(video)
            video_batch = video.to(self.device)
            batch_features = self.video_feat_extractor(video_batch)
            batch_features = F.normalize(batch_features, dim=1)
            features = batch_features.cpu().numpy()
        return features

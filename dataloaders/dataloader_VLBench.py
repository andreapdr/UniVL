import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)

import sys

sys.path.append(parent_dir)

from math import ceil, floor
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
        task="action",
        device="cpu",
        centercrop=True,
        process_at_train=False,
    ):
        self._data = json.load(open(datapath))
        self.data = list(self._data.values())
        self.task = task
        self.tokenizer = tokenizer
        self.video_feat_extractor = video_feature_extractor
        self.video_preprocessor = Preprocessing(
            type="s3dg", FRAMERATE_DICT={"s3dg": 16}
        )
        self.videodir = os.path.expanduser(videodir)
        self.device = device
        self.centercrop = centercrop
        self.process_at_train = process_at_train

    def __len__(self):
        return len(self.data)

    def _get_text(self, idx):
        _capt = self.data[idx]["caption"]
        _foil = self.data[idx]["foils"][0]
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
        if self.process_at_train:
            video_feat = self._extract_features(video_id)
        else:
            if self.data[video_id]["youtube_id"] is not None:
                video_fname = self.data[video_id]["youtube_id"]
            else:
                video_fname = self.data[video_id]["video_file"]
            feature_path = os.path.join("cache", "processed", video_fname + ".npy")
            video_feat = np.load(feature_path)
        video_mask = torch.ones(size=(1, video_feat.shape[0]))  # TODO: check this
        return {"video_feat": video_feat, "video_mask": video_mask}

    def __getitem__(self, idx):
        text = self._get_text(idx)
        video = self._get_video(idx)
        video_id = self.data[idx]["dataset_idx"]
        return video, text, video_id

    def _extract_features(self, idx):
        if self.data[idx]["youtube_id"] is not None:
            video_fname = self.data[idx]["youtube_id"] + ".mp4"
        else:
            video_fname = self.data[idx]["video_file"]
            if self.data[idx]["dataset"] == "smsm":
                video_fname += ".webm"
            elif self.data[idx]["dataset"] == "ikea":
                video_fname += ".avi"
            else:
                video_fname += ".mp4"

        video_path = os.path.join(self.videodir, video_fname)
        start_time = self.data[idx]["start_time"]
        end_time = self.data[idx]["end_time"]

        video = self._load_video(video_path, start_time, end_time)

        if self.centercrop:
            video = torchvision.transforms.CenterCrop((224, 224))(video)

        with torch.no_grad():
            self.video_feat_extractor.eval()
            video = video.squeeze()
            # Batch x 3 x T x H x W
            video = self.video_preprocessor(video)
            video = video.to(self.device)
            video = self.video_feat_extractor(video)
            video = F.normalize(video, dim=1)
            # video = video.cpu().numpy()
        return video

    def _load_video(Self, video_path, start_time=None, end_time=None):
        if start_time is None:
            return torchvision.io.read_video(video_path, pts_unit="sec")[0].permute(
                0, 3, 1, 2
            )
        else:
            return torchvision.io.read_video(
                video_path,
                pts_unit="sec",
                start_pts=floor(start_time),
                end_pts=ceil(end_time),
            )[0].permute(0, 3, 1, 2)

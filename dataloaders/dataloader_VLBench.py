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
import ffmpeg


class VLBenchDataset(Dataset):
    def __init__(
        self,
        datapath,
        tokenizer,
        video_feature_extractor,
        videodir="~/datasets/vl-bench/videos",
        device="cpu",
        centercrop=True,
        process_at_train=False,
    ):
        self._data = json.load(open(datapath))
        self.data = list(self._data.values())
        self.tokenizer = tokenizer
        self.video_feat_extractor = video_feature_extractor
        self.video_preprocessor = Preprocessing(
            type="s3dg", FRAMERATE_DICT={"s3dg": 16}
        )
        self.videodir = os.path.expanduser(videodir)
        self.device = device
        self.centercrop = centercrop
        self.process_at_train = process_at_train
        if "change-state" in datapath:
            self.cache_dir = os.path.join("cache", "change-state", "processed")
        else:
            self.cache_dir = os.path.join(
                "cache", datapath.split("/")[-1].split(".")[0], "processed"
            )

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

    def _format_filename(self, video_id, start_time, end_time):
        if end_time == -1 or end_time is None:
            return video_id
        else:
            return f"{video_id}_{int(start_time)}_{int(end_time)}"

    def _get_video(self, video_id):
        if self.process_at_train:
            # video_feat = self._extract_features(video_id)
            video_feat = self._extract_features(video_id)
        else:
            if self.data[video_id]["youtube_id"] is not None:
                video_fname = self.data[video_id]["youtube_id"]
            else:
                video_fname = self.data[video_id]["video_file"]
            # are we loading the whole video or a clip?
            start_time = self.data[video_id]["start_time"]
            end_time = self.data[video_id]["end_time"]

            feature_filename = self._format_filename(video_fname, start_time, end_time)
            feature_path = os.path.join(self.cache_dir, feature_filename + ".npy")
            video_feat = np.load(feature_path)
        video_mask = torch.ones(size=(1, video_feat.shape[0]))  # TODO: check this
        return {"video_feat": video_feat, "video_mask": video_mask}

    def __getitem__(self, idx):
        text = self._get_text(idx)
        video = self._get_video(idx)
        video_id = self.data[idx]["dataset_idx"]
        return video, text, video_id, str(idx)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        return height, width

    def _get_output_dim(self, h, w):
        if isinstance(224, tuple) and len(224) == 2:
            return 224
        elif h >= w:
            return int(h * 224 / w), 224
        else:
            return 224, int(w * 224 / h)

    def _extract_features(self, video_id):
        start_time = self.data[video_id]["start_time"]
        end_time = self.data[video_id]["end_time"]

        video_path = self.data[video_id]["youtube_id"]

        if video_path is None:
            video_path = self.data[video_id]["video_file"]

        video_path = os.path.join(self.videodir, video_path)

        if self.data[video_id]["dataset"] == "something-something-v2":
            video_path += ".webm"
        else:
            video_path += ".mp4"

        if end_time != -1 and end_time is not None:
            # we have to cut the video and save it to a temporary file
            _cut_video = torchvision.io.read_video(
                video_path,
                pts_unit="sec",
                start_pts=floor(start_time),
                end_pts=ceil(end_time),
            )
            torchvision.io.write_video(
                filename="cache/process_runtime/tmp.mp4",
                video_array=_cut_video[0],
                fps=_cut_video[-1]["video_fps"],
            )
            video_path = "cache/process_runtime/tmp.mp4"

        h, w = self._get_video_dim(video_path)

        height, width = self._get_output_dim(h, w)
        cmd = (
            ffmpeg.input(video_path)
            .filter("fps", fps=16)
            .filter("scale", width, height)
        )
        if self.centercrop:
            x = int((width - 224) / 2.0)
            y = int((height - 224) / 2.0)
            cmd = cmd.crop(x, y, 224, 224)
        out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
            capture_stdout=True, quiet=True
        )
        if self.centercrop and isinstance(224, int):
            height, width = 224, 224
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        video = torch.from_numpy(video.astype("float32"))
        video = video.permute(0, 3, 1, 2)

        with torch.no_grad():
            self.video_feat_extractor.eval()
            video = video.squeeze()
            # Batch x 3 x T x H x W
            video = self.video_preprocessor(video)
            video = video.to(self.device)
            video = self.video_feat_extractor(video)
            video = F.normalize(video, dim=1)
            video = video.to("cpu").numpy()
        return video

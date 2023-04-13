import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
from modules.modeling import UniVL
from modules.tokenization import BertTokenizer


class FOIL_CONFIG:
    def __init__(self):
        self.max_words = 20
        self.max_frames = 100
        self.stage_two = False
        self.video_dim = 1024
        self.batch_size = 1
        self.n_gpu = 1
        self.n_pair = 1
        self.margin = 0.1
        self.negative_weighting = 1
        self.hard_negative_rate = 0.5
        self.use_mil = False
        self.local_rank = 0


def init_univl(device):
    model_state_dict = torch.load(
        "../UniVL/weight/univl.pretrained.bin", map_location="cpu"
    )
    task_config = FOIL_CONFIG()

    cache_dir = None
    model = UniVL.from_pretrained(
        pretrained_bert_name="bert-base-uncased",
        visual_model_name="visual-base",
        cross_model_name="cross-base",
        decoder_model_name="decoder-base",
        state_dict=model_state_dict,
        cache_dir=cache_dir,
        task_config=task_config,
    )
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    return model, tokenizer


def run_preprocessing(
    input_file,
    something_something_dir,
    coin_dir,
    youcook2_dir,
    star_dir,
    rareact_dir,
    cache_dir="./cache",
):
    from VideoFeatureExtractor.preprocess_vlbench import convert_multi3bench
    from VideoFeatureExtractor.extract import extract

    dataset_path = os.path.expanduser(input_file)
    dataset_name = dataset_path.split("/")[-1].split(".")[0]
    if "change-state" in dataset_name:
        dataset_name = "change-state"

    video_dirs = {
        "youcook2": youcook2_dir,
        "something-something-v2": something_something_dir,
        "coin": coin_dir,
        "star": star_dir,
        "rareact": rareact_dir,
    }

    cache_dir = f"{cache_dir}/{dataset_name}"
    os.makedirs(cache_dir, exist_ok=True)
    output_path = f"{cache_dir}/vlbench_s3dg.csv"
    convert_multi3bench(dataset_path, output_path, cache_dir, video_dirs)
    extract(
        dataset_path=output_path,
        batch_size=1,
        num_decoding_thread=1,
        debug=False,
        model_path="../UniVL/VideoFeatureExtractor/model/s3d_howto100m.pth",
    )
    return

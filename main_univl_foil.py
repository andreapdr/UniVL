import torch
import os
from modules.modeling import UniVL
from argparse import ArgumentParser
from modules.tokenization import BertTokenizer
from tqdm import tqdm
from dataloaders.dataloader_VLBench import VLBenchDataset
from VideoFeatureExtractor.model import get_s3dg_model
import json


def init_model(args, device):
    model_state_dict = torch.load("weight/univl.pretrained.bin", map_location="cpu")

    cache_dir = None
    model = UniVL.from_pretrained(
        pretrained_bert_name="bert-base-uncased",
        visual_model_name="visual-base",
        cross_model_name="cross-base",
        decoder_model_name="decoder-base",
        state_dict=model_state_dict,
        cache_dir=cache_dir,
        task_config=args,
    )

    model.to(device)

    return model


def run_preprocessing(args):
    from VideoFeatureExtractor.preprocess_vlbench import convert_multi3bench
    from VideoFeatureExtractor.extract import extract

    # print("- extracting video features via S3DG")
    dataset_path = os.path.expanduser(args.json_path)
    dataset_name = dataset_path.split("/")[-1].split(".")[0]
    if "change-state" in dataset_name:
        # to avoid recomputing video feats for the different settings
        dataset_name = "change-state"
    # video_dir = os.path.expanduser(args.video_dir)
    video_dirs = {
        "youcook2": os.path.expanduser("~/datasets/vl-bench/videos/youcook2"),
        "rareact": os.path.expanduser("~/datasets/vl-bench/videos/rareact"),
        "coin": os.path.expanduser("~/datasets/vl-bench/videos/coin"),
        "something-something-v2": os.path.expanduser(
            "~/datasets/vl-bench/videos/something-something-v2"
        ),
        "star": os.path.expanduser("~/datasets/vl-bench/videos/star"),
    }
    cache_path = f"cache/{dataset_name}"
    os.makedirs(cache_path, exist_ok=True)
    output_path = f"{cache_path}/vlbench_s3dg.csv"
    # print(f"- video directory: {video_dir}\n- cache dir: {cache_path}")
    convert_multi3bench(dataset_path, output_path, cache_path, video_dirs)
    extract(
        dataset_path=output_path, batch_size=1, num_decoding_thread=1, debug=args.debug
    )
    return


def run_vlbench(args):
    print(f"- evaluating on: {args.json_path}")
    if not args.process_at_train:
        run_preprocessing(args)
    device = args.device
    benchmark_path = os.path.expanduser(args.json_path)
    model = init_model(args, device=device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # print("- loaded pre-trained UniVL model")
    s3dg_model = get_s3dg_model(
        s3dg_path="VideoFeatureExtractor/model/s3d_howto100m.pth", device=device
    )
    # print("- loaded pre-trained S3DG video-feature extractor")

    vldataset = VLBenchDataset(
        datapath=benchmark_path,
        tokenizer=tokenizer,
        video_feature_extractor=s3dg_model,
        videodir=args.video_dir,
        device=device,
        process_at_train=True if args.process_at_train else False,
    )

    results = {}

    dataloader = torch.utils.data.DataLoader(vldataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        pairwise_acc = 0
        for batch in tqdm(dataloader):
            video, _text, video_id, ann_id = batch
            if args.debug:
                print(video_id[0])
            video_feat = video["video_feat"].to(device)
            video_mask = video["video_mask"].to(device)
            capt_id = _text["capt"].to(device)
            capt_mask = _text["capt_mask"].to(device)
            capt_token_type = _text["capt_token_type"].to(device)
            foil_id = _text["foil"].to(device)
            foil_mask = _text["foil_mask"].to(device)
            foil_token_type = _text["foil_token_type"].to(device)

            capt_sequence_output, capt_visual_output = model.get_sequence_visual_output(
                capt_id, capt_mask, capt_token_type, video_feat, video_mask
            )
            foil_sequence_output, foil_visual_output = model.get_sequence_visual_output(
                foil_id, foil_mask, foil_token_type, video_feat, video_mask
            )

            capt_sim = model.get_similarity_logits(
                capt_sequence_output, capt_visual_output, capt_mask, video_mask
            )
            foil_sim = model.get_similarity_logits(
                foil_sequence_output, foil_visual_output, foil_mask, video_mask
            )
            if capt_sim > foil_sim:
                pairwise_acc += 1

            results[ann_id[0]] = {"scores": [capt_sim.item(), foil_sim.item()]}

        print(f"- Pairwise accuracy: {pairwise_acc / len(vldataset):.3f}")

    json.dump(results, open(f"results_{args.json_path.split('/')[-1]}.json", "w"))


if __name__ == "__main__":
    parser = ArgumentParser(description="UniVL on VL-Bench")
    parser.add_argument(
        "--json_path",
        type=str,
        default="~/datasets/vl-bench/annotations/change-state-action.json",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="~/datasets/vl-bench/videos/change-state",
    )
    parser.add_argument("--max_words", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--stage_two", action="store_true")
    parser.add_argument(
        "--video_dim", type=int, default=1024, help="video feature dimension"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument(
        "--n_pair", type=int, default=1, help="Num of pair to output from data loader"
    )
    parser.add_argument("--margin", type=float, default=0.1, help="margin for loss")
    parser.add_argument(
        "--negative_weighting",
        type=int,
        default=1,
        help="Weight the loss for intra negative",
    )
    parser.add_argument(
        "--hard_negative_rate",
        type=float,
        default=0.5,
        help="rate of intra negative sample",
    )
    parser.add_argument(
        "--use_mil", action="store_true", help="Whether use MIL as Miech et. al. (2020)"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--process_at_train",
        action="store_true",
        help="Extract video features at run time instead of pre-processing them before inference",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_vlbench(args)

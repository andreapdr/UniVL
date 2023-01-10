from util import get_logger
from modules.tokenization import BertTokenizer
from modules.modeling import UniVL
from dataloaders.dataloader_youcook_retrieval import Youcook_DataLoader
from dataloaders.dataloader_multi3bench import Multi3bench_DataLoader
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import numpy as np
import json
import random
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'    # TODO


def get_args(description='UniVL on Pretrain'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-file", required=True,
                        help="Path to annotations.json")
    parser.add_argument("--video-feature", required=True,
                        help="Path to video feature directory")
    parser.add_argument("--change-state-setting", default="action_foil",
                        help="Setting of the instrument Change-of-State in the Multi3bench")
    parser.add_argument("--instrument", default="change-state",
                        help="Multi3bench's instrument to test")

    parser.add_argument("--do_pretrain", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str,
                        default='data/HowTo100M_v1.csv', help='train csv')
    parser.add_argument('--features_path', type=str,
                        default='feature', help='feature path for 2D features')
    parser.add_argument('--data_path', type=str,
                        default='data/data.pickle', help='data pickle file path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int,
                        default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100,
                        help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024,
                        help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--min_words', type=int, default=0, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float,
                        default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float,
                        default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float,
                        default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int,
                        default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1,
                        help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False,
                        help="Bert pre-trained model")
    parser.add_argument("--visual_model", default="visual-base",
                        type=str, required=False, help="Visual module")
    parser.add_argument("--cross_model", default="cross-base",
                        type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base",
                        type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default="weight/univl.pretrained.bin", type=str,
                        required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1,
                        help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--world_size", default=0,
                        type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0,
                        type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1,
                        help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true',
                        help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true',
                        help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int,
                        default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int,
                        default=6, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int,
                        default=2, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers',
                        type=int, default=3, help="Layer NO. of decoder.")

    parser.add_argument('--stage_two', action='store_true',
                        help="Whether training with decoder.")
    parser.add_argument('--pretrain_enhance_vmodal', action='store_true',
                        help="Enhance visual and other modalities when pretraining.")

    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_model", default="pytorch_model.bin.checkpoint", type=str, required=False,
                        help="Save the last model as a checkpoint.")

    args = parser.parse_args()

    if args.sampled_use_mil:  # sample from each video, has a higher priority than use_mil.
        args.use_mil = True

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    args.checkpoint_model = '{}_{}_{}_{}.checkpoint'.format(
        args.checkpoint_model, args.bert_model, args.max_words, args.max_frames)

    return args


def init_model(args, device):
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location="cpu")
    else:
        model_state_dict = None

    cache_dir = None
    model = UniVL.from_pretrained(
        args.bert_model,
        args.visual_model,
        args.cross_model,
        args.decoder_model,
        cache_dir=cache_dir,
        state_dict=model_state_dict,
        task_config=args
    )

    model.to(device)

    return model


def compute_results(preds, labels):
    pass


def main():
    args = get_args()
    device = "cuda"

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model,
        do_lower_case=True)

    model = init_model(args, device)

    dataset_multi3bench = Multi3bench_DataLoader(
        tokenizer,
        datapath=args.input_file,
        feature_dir=args.video_feature,
        instrument=args.instrument
    )

    dataloader_multi3bench = DataLoader(
        dataset_multi3bench,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )

    model.eval()
    results = defaultdict(dict)
    with torch.no_grad():
        for bid, (batch, _) in enumerate(tqdm(dataloader_multi3bench)):
            texts, video, video_mask = batch
            for task_id, task_data in texts.items():
                texts_batch = tuple(t.to(device) for t in task_data)
                text_id, text_mask, text_segment = texts_batch
                video = video.to(device)
                video_mask = video_mask.to(device)

                sequence_output, visual_output = model.get_sequence_visual_output(
                    text_id,
                    text_mask,
                    text_segment,
                    video,
                    video_mask
                )

                sim = model.get_similarity_logits(
                    sequence_output,
                    visual_output,
                    text_mask,
                    video_mask
                )

                results[bid][task_id] = sim.item()

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    json.dump(results, open(f"results/results_{now}.json", "w"))


if __name__ == "__main__":
    main()

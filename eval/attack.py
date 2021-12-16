import torch
# import torch.nn as nn
import torch.nn.functional as F

from fairseq import (
    checkpoint_utils,
    tasks,
    utils,
)
from fairseq.data.audio.speech_to_text_dataset import get_features_or_waveform
from fairseq.data.audio.feature_transforms import (
    CompositeAudioFeatureTransform,
)
from utils import (
    UtteranceCMVNPyTorch,
    FBankPyTorch
)
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset

import logging
from tqdm.auto import tqdm

# general
import math
# import shutil
# import numpy as np
from pathlib import Path
import soundfile as sf
import sys
import argparse
import glob


# local
logger = logging.getLogger(__name__)

cuclear = torch.cuda.empty_cache


@torch.no_grad()
def clamp(x_adv, x, epsilon, margin=1e-6):
    tmp = x + (epsilon - margin)
    x_adv = torch.where(x_adv > tmp, tmp, x_adv)
    tmp = x - (epsilon - margin)
    x_adv = torch.where(x_adv < tmp, tmp, x_adv)
    return x_adv


@torch.no_grad()
def zero_one_clamp(x, mean=0, std=1, margin=1e-6):
    x = torch.clamp(
        x * std + mean,
        min=margin,
        max=1 - margin
    )
    return (x - mean) / std


def ensemble_logits(logits):
    """ expects list of unnormalized logits """
    N = len(logits)
    log_probs = [logit.log_softmax(-1) for logit in logits]
    avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(N)
    return avg_probs


def calc_db(x_adv, x, margin=1e-8):
    def f_db(z):
        return math.log10(z.abs().max() + margin) * 20
    return f_db(x_adv - x) - f_db(x)


def targeted_fn(images, labels):
    return (labels + 1) % 100


def mifgsm(models, input_transform, sample, loss_fn, epsilon, alpha, num_iter, random_start, decay):
    """ x is waveform, y is sequence """
    x = sample["net_input"]["src_tokens"]
    x_len = sample["net_input"]["src_lengths"]
    prev_y = sample["net_input"]["prev_output_tokens"]
    y = sample["target"]
    bsz = y.size(0)
    blank_idx = 0
    pad_idx = 1
    eos_idx = 2

    bar = tqdm(range(num_iter), desc="attacking", leave=False)

    x_adv = x.clone().detach()
    if random_start:
        # Starting at a uniformly random point
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = zero_one_clamp(x_adv).detach()

    momentum = torch.zeros_like(x).detach().type_as(x)

    def _collate(inputs):
        frames = []
        src_lengths = []
        # input transform (e.g. fbank & ucmvn)
        for wav, slen in zip(inputs, x_len):
            fbank = input_transform(wav[:slen].unsqueeze(0))
            frames.append(fbank)
            src_lengths.append(fbank.size(0))

        # collate
        max_len = max(src_lengths)
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
        for i, v in enumerate(frames):
            out[i, : v.size(0)] = v
        return out, torch.LongTensor(src_lengths).type_as(prev_y)

    for _ in bar:
        x_adv.requires_grad = True
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        src_tokens, src_lengths = _collate(x_adv)

        logits = []
        for model in models:
            model.zero_grad()
            logit, extra = model(src_tokens, src_lengths, prev_output_tokens=prev_y)
            logits.append(logit)

        # reshape lprobs to (L,B,X) for torch.ctc
        lprobs = ensemble_logits(logits)
        max_src = lprobs.size(1)
        lprobs = lprobs.transpose(1, 0).contiguous()

        # get subsampling padding mask & lengths
        if extra["padding_mask"] is not None:
            non_padding_mask = ~extra["padding_mask"]
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = lprobs.new_ones(
                (bsz, max_src), dtype=torch.long).sum(-1)

        pad_mask = (y != pad_idx) & (
            y != eos_idx
        )
        targets_flat = y.masked_select(pad_mask)
        target_lengths = pad_mask.long().sum(-1)

        if loss_fn == "xent":
            loss = F.cross_entropy(
                lprobs.view(bsz * lprobs.size(1), -1),
                y.view(-1),
                reduction="mean",
            )
        elif loss_fn == "ctc":
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=blank_idx,
                reduction="mean",
                zero_infinity=True,
            )
        elif loss_fn == "ctc_improved":
            raise NotImplementedError(loss_fn)
        else:
            raise NotImplementedError(loss_fn)

        loss.backward()
        # loss.backward(retain_graph=True)

        grad = x_adv.grad.detach()  # (bsz, length)
        grad = grad / torch.mean(torch.abs(grad), dim=(1,), keepdim=True)
        grad = grad + momentum * decay
        momentum = grad

        x_adv = x_adv + alpha * grad.sign()
        x_adv = clamp(x_adv, x, epsilon)

        bar.set_postfix(loss=loss.item())
    return x_adv


class Attacker:

    def __init__(self, args):
        self.args = args
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level="INFO",  # "DEBUG" "WARNING" "ERROR"
            stream=sys.stdout,
        )
        # logger = logging.getLogger(__name__)

        self.batch_size = args.batch_size
        self.device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

        self.epsilon = args.epsilon / (2**15)
        self.alpha = self.epsilon / 10.

    def load_models(self, args):
        logger.info("loading model(s) from {}".format(args.model_dir))

        all_config = []
        all_models = []
        for checkpoint in glob.glob(f"{args.model_dir}/*.pt"):
            states = checkpoint_utils.load_checkpoint_to_cpu(
                path=checkpoint, arg_overrides=None, load_on_all_ranks=False)
            cfg = states["cfg"]
            cfg.task.config_yaml = args.config_yaml
            cfg.common.user_dir = args.user_dir
            utils.import_user_module(cfg.common)

            # Setup task, e.g., translation, language modeling, etc.
            task = tasks.setup_task(cfg.task)
            model = task.build_model(cfg.model)
            model.load_state_dict(
                states["model"], strict=True, model_cfg=cfg.model
            )
            model.to(self.device)
            model.prepare_for_inference_(cfg)
            all_config.append(cfg)
            all_models.append(model)
            logger.info("loaded model from {}".format(checkpoint))

        return all_models, all_config

    def load_data(self, args, task):
        with open(args.wav_list, "r") as f:
            lines = [line.strip() for line in f]
        with open(args.text, "r") as f:
            tgt_texts = [line.strip() for line in f]
        lengths = [get_features_or_waveform(p).shape[0] for p in lines]
        dataset = SpeechToTextDataset(
            "attack", False, task.data_cfg,
            lines, lengths, tgt_texts=tgt_texts, tgt_dict=task.tgt_dict,
            pre_tokenizer=task.build_tokenizer(None),
            bpe_tokenizer=task.build_bpe(None),
        )
        data_iter = task.get_batch_iterator(
            dataset=dataset,
            max_sentences=self.batch_size,
            max_positions=60000
        )
        # usage: itr = data_iter.next_epoch_itr(shuffle=False)
        return data_iter, dataset

    def solve(self,):

        args = self.args
        models, cfgs = self.load_models(args)
        task = tasks.setup_task(cfgs[0].task)  # includes dictionary, tokenizer, bpe information
        epochs_iter, benign_set = self.load_data(args, task)

        input_transform = CompositeAudioFeatureTransform([
            FBankPyTorch(denormalize=True, num_mel_bins=80, sample_rate=48000),
            UtteranceCMVNPyTorch(norm_means=True, norm_vars=True)
        ])
        logger.info(input_transform)

        adv_wavs = []
        adv_ids = []
        total_dBs = []
        bar = tqdm(epochs_iter.next_epoch_itr(shuffle=False), desc="sample")
        for sample in bar:
            sample = utils.move_to_cuda(sample)
            wav = mifgsm(
                models,
                input_transform=input_transform,
                sample=sample,
                loss_fn="ctc",
                epsilon=self.epsilon,
                alpha=self.alpha,
                num_iter=self.args.num_iter,
                random_start=self.args.random_start,
                decay=self.args.decay
            )
            wav = wav.detach().cpu()
            benign = sample["net_input"]["src_tokens"].cpu()
            error = (wav - benign).abs()
            assert not (error > self.epsilon).any()
            dBs = [calc_db(a, b) for a, b in zip(wav, benign)]
            total_dBs += dBs
            bar.set_postfix(dB=sum(dBs) / len(dBs))
            adv_wavs.extend(wav)
            adv_ids.extend(sample["id"].tolist())

        logger.info(f"Average distortion (dB): {sum(total_dBs)/len(total_dBs):.3f}")

        if self.args.savedir is not None:
            adv_wavs = [a for a, b in sorted(zip(adv_wavs, adv_ids), key=lambda x: x[1])]
            adv_ids = sorted(adv_ids)
            logger.info("validating wave constraints...")

            output = Path(self.args.savedir).absolute()
            output.mkdir(exist_ok=True)
            (output / "wav").mkdir(exist_ok=True)
            f_wav_list = open(output / "adv.wav_list", "w")

            sample_rate = 48000
            for i, (waveform, utt_id) in enumerate(zip(adv_wavs, adv_ids)):

                wav_path = (output / "wav" / f"{utt_id}.wav").as_posix()
                sf.write(
                    wav_path,
                    waveform.squeeze(0).cpu().numpy(),
                    samplerate=int(sample_rate)
                )
                f_wav_list.write(str(wav_path) + "\n")

            f_wav_list.close()

    #     self.validate(adv_wavs, benign_set)

    # def validate(self, adv_wavs, benign_set):
    #     assert len(adv_wavs) == len(benign_set)
    #     benign_wavs = [d.source for d in benign_set]
    #     dBs = []
    #     for adv, benign in zip(adv_wavs, benign_wavs):
    #         error = (adv - benign).abs()
    #         assert not (error > self.epsilon).any()
    #         dBs.append(calc_db(adv, benign))

    #     logger.info(f"Average distortion (dB): {sum(dBs)/len(dBs):.3f}")

    @staticmethod
    def add_args(parser):
        # fairseq
        parser.add_argument("--user-dir", default="../codebase")
        parser.add_argument("--config-yaml", default="./config_attack.yaml")
        # data
        parser.add_argument("--wav-list", default="./data/test2.wav_list")
        parser.add_argument("--text", default="./data/test2.en")
        # parser.add_argument("--num-workers", type=int, default=2)
        parser.add_argument("--epsilon", type=float, default=400,
                            help="epsilon for Linf for waveform assuming 16-bit integer")
        # training
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--cpu", action="store_true")
        # checkpoint
        parser.add_argument("--model-dir", default="./models")

        parser.add_argument("--iters", type=int, dest="num_iter", default=1)

        parser.add_argument("--targeted", action="store_true")
        parser.add_argument("--decay", type=float, default=0.0)
        parser.add_argument("--random-start", action='store_true')

        parser.add_argument("--savedir", default=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Attacker.add_args(parser)
    args = parser.parse_args()

    Attacker(args).solve()

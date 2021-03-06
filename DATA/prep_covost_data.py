#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple
import numpy as np

import pandas as pd
import torchaudio
from data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
    cal_gcmvn_stats,
)
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive
from tqdm import tqdm
import jieba_fast as jieba

from text_processors import TextNormalizer
log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]


class CoVoST(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).

    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str, optional): target (text) language,
        None for no translation (default: None)
        version (int, optional): CoVoST version. (default: 2)
        download (bool, optional): Whether to download the dataset if it is not
        found at root path. (default: ``False``).
    """

    COVOST_URL_TEMPLATE = (
        "https://dl.fbaipublicfiles.com/covost/"
        "covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
    )

    VERSIONS = {2}
    SPLITS = ["train", "dev", "test"]

    XX_EN_LANGUAGES = {
        1: ["fr", "de", "nl", "ru", "es", "it", "tr", "fa", "sv-SE", "mn", "zh-CN"],
        2: [
            "fr",
            "de",
            "es",
            "ca",
            "it",
            "ru",
            "zh-CN",
            "pt",
            "fa",
            "et",
            "mn",
            "nl",
            "tr",
            "ar",
            "sv-SE",
            "lv",
            "sl",
            "ta",
            "ja",
            "id",
            "cy",
        ],
    }
    EN_XX_LANGUAGES = {
        1: [],
        2: [
            "de",
            "tr",
            "fa",
            "sv-SE",
            "mn",
            "zh-CN",
            "cy",
            "ca",
            "sl",
            "et",
            "id",
            "ar",
            "ta",
            "lv",
            "ja",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: Optional[str] = None,
        version: int = 2,
    ) -> None:
        assert version in self.VERSIONS and split in self.SPLITS
        assert source_language is not None
        self.no_translation = target_language is None
        if not self.no_translation:
            assert "en" in {source_language, target_language}
            if source_language == "en":
                assert target_language in self.EN_XX_LANGUAGES[version]
            else:
                assert source_language in self.XX_EN_LANGUAGES[version]
        else:
            # Hack here so that we can get "split" column from CoVoST TSV.
            # Note that we use CoVoST train split for ASR which is an extension
            # to Common Voice train split.
            target_language = "de" if source_language == "en" else "en"

        self.root: Path = Path(root)

        cv_tsv_path = self.root / "validated.tsv"
        assert cv_tsv_path.is_file()

        covost_url = self.COVOST_URL_TEMPLATE.format(
            src_lang=source_language, tgt_lang=target_language
        )
        covost_archive = self.root / Path(covost_url).name
        if not covost_archive.is_file():
            download_url(covost_url, self.root.as_posix(), hash_value=None)
        extract_archive(covost_archive.as_posix())

        cv_tsv = load_df_from_tsv(cv_tsv_path)
        covost_tsv = load_df_from_tsv(
            self.root / Path(covost_url).name.replace(".tar.gz", "")
        )
        df = pd.merge(
            left=cv_tsv[["path", "sentence", "client_id"]],
            right=covost_tsv[["path", "translation", "split"]],
            how="inner",
            on="path",
        )
        if split == "train":
            df = df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
        else:
            df = df[df["split"] == split]
        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        to_rm = re.compile(r"\[?(TO\s)?REMOVE\]?")
        for e in tqdm(data, desc="verify"):
            try:
                path = self.root / "clips" / e["path"]
                _ = torchaudio.info(path.as_posix())
                e["translation"] = to_rm.sub("", e["translation"]).strip()
                if len(e["translation"]) == 0:
                    continue
                self.data.append(e)
            except RuntimeError:
                pass

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, Optional[str], str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, sentence, translation, speaker_id,
            sample_id)``
        """
        data = self.data[n]
        path = self.root / "clips" / data["path"]
        waveform, sample_rate = torchaudio.load(path)
        sentence = data["sentence"]
        translation = None if self.no_translation else data["translation"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        return waveform, sample_rate, sentence, translation, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute() / args.src_lang
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")
    # Extract features
    feature_root = root / "fbank80"
    datasets = {split: CoVoST(root, split, args.src_lang, args.tgt_lang) for split in CoVoST.SPLITS}
    if not args.manifest_only:
        feature_root.mkdir(exist_ok=True)
        for split in CoVoST.SPLITS:
            print(f"Fetching split {split}...")
            dataset = datasets[split]
            print("Extracting log mel filter bank features...")
            gcmvn_feature_list = []
            if split == 'train' and args.cmvn_type == "global":
                print("And estimating cepstral mean and variance stats...")
            for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset, desc="fbank"):
                features = extract_fbank_features(
                    waveform, sample_rate, feature_root / f"{utt_id}.npy", overwrite=True
                )
                if split == 'train' and args.cmvn_type == "global":
                    if len(gcmvn_feature_list) < args.gcmvn_max_num:
                        gcmvn_feature_list.append(features)

            if split == 'train' and args.cmvn_type == "global":
                # Estimate and save cmv
                stats = cal_gcmvn_stats(gcmvn_feature_list)
                with open(root / "gcmvn.npz", "wb") as f:
                    np.savez(f, mean=stats["mean"], std=stats["std"])

    # Pack features into ZIP
    zip_path = root / "fbank80.zip"
    if not args.manifest_only:
        print("ZIPing features...")
        create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text_src = []
    train_text_tgt = []
    src_normalizer = TextNormalizer(args.src_lang)
    tgt_normalizer = TextNormalizer(args.tgt_lang)
    task = f"asr_{args.src_lang}"
    if args.tgt_lang is not None:
        task = f"st_{args.src_lang}_{args.tgt_lang}"
    for split in CoVoST.SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = datasets[split]
        for _, _, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset, desc="manifest"):
            manifest["id"].append(utt_id)
            manifest["audio"].append(audio_paths[utt_id])
            manifest["n_frames"].append(audio_lengths[utt_id])
            src_utt = src_normalizer(src_utt)
            if args.src_lang == "zh-CN" and args.jieba:
                src_utt = " ".join(jieba.cut(src_utt, cut_all=False))
                src_utt = re.sub(r"\s\s+", " ", src_utt)
            manifest["src_text"].append(src_utt)
            tgt_utt = tgt_normalizer(tgt_utt)
            if args.tgt_lang == "zh-CN" and args.jieba:
                tgt_utt = " ".join(jieba.cut(tgt_utt, cut_all=False))
                tgt_utt = re.sub(r"\s\s+", " ", tgt_utt)
            manifest["tgt_text"].append(tgt_utt)
            manifest["speaker"].append(speaker_id)
        is_train_split = split.startswith("train")
        if is_train_split:
            train_text_tgt.extend(manifest["tgt_text"])
            train_text_src.extend(manifest["src_text"])

        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split)
        save_df_to_tsv(df, root / f"{split}_{task}.tsv")
    # Generate vocab
    vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}_st_{args.src_lang}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text_src:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size
        )
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}_st_{args.tgt_lang}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text_tgt:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size
        )
    # Generate config YAML
    gen_config_yaml(
        root,
        spm_filename=spm_filename_prefix + ".model",
        yaml_filename=f"config_{task}.yaml",
        specaugment_policy="lb",
        cmvn_type=args.cmvn_type,
        gcmvn_path=(
            root / "gcmvn.npz" if args.cmvn_type == "global"
            else None
        ),
    )
    # Clean up
    if not args.manifest_only:
        shutil.rmtree(feature_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=5000, type=int)
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--tgt-lang", "-t", type=str)
    parser.add_argument("--manifest-only", action="store_true", help="only preprocess manifest. "
                        "works only when zip is already present"
                        )
    parser.add_argument("--jieba", action="store_true", help="use jieba for chinese.")
    parser.add_argument(
        "--cmvn-type", default="utterance",
        choices=["global", "utterance"],
        help="The type of cepstral mean and variance normalization"
    )
    parser.add_argument(
        "--gcmvn-max-num", default=25000, type=int,
        help="Maximum number of sentences to use to estimate global mean and variance"
    )
    args = parser.parse_args()
    print(args)
    process(args)


if __name__ == "__main__":
    main()

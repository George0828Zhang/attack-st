#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Tuple
# from collections import OrderedDict
# import re
# import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from fairseq import checkpoint_utils
# from fairseq.data.data_utils import lengths_to_padding_mask


from fairseq.models import (
    # FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
# from fairseq.modules import LayerNorm
from fairseq.models.speech_to_text.s2t_transformer import (
    S2TTransformerEncoder,
    s2t_transformer_s,
)

# user
from .nat_utils import generate

logger = logging.getLogger(__name__)


@register_model("s2t_speech_encoder")
class SpeechEncoderModel(FairseqEncoderModel):
    """
    causal encoder + output projection
    """
    def __init__(self, encoder):
        super().__init__(encoder)
        self.one_pass_decoding = True  # must implement generate()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(SpeechEncoderModel, SpeechEncoderModel).add_args(parser)
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )

    @classmethod
    def build_encoder(cls, args, src_dict, ctc_projection):
        encoder = SpeechEncoder(args, src_dict, ctc_projection)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        src_dict = task.source_dictionary
        ctc_projection = nn.Linear(
            args.encoder_embed_dim,
            len(src_dict),
            bias=False
        )
        nn.init.normal_(
            ctc_projection.weight, mean=0, std=args.encoder_embed_dim ** -0.5
        )
        encoder = cls.build_encoder(
            args, src_dict, ctc_projection)
        return cls(encoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        logits = net_output[0]

        if torch.is_tensor(logits):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits_f = logits.float()
            if log_probs:
                lprobs = F.log_softmax(logits_f, dim=-1)
            else:
                lprobs = F.softmax(logits_f, dim=-1)
        else:
            raise NotImplementedError

        return lprobs

    def forward(
        self, src_tokens, src_lengths,
        return_all_hiddens: bool = False,
        **unused,
    ):
        encoder_out = self.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens
        )
        x = encoder_out["encoder_out"][0]  # speech hidden states
        x = self.encoder.ctc_projection(x)  # ctc projection
        x = x.transpose(1, 0)  # force batch first

        padding_mask = encoder_out["encoder_padding_mask"][0] \
            if len(encoder_out["encoder_padding_mask"]) > 0 else None
        extra = {
            "padding_mask": padding_mask,
            "encoder_out": encoder_out
        }
        return x, extra

    @property
    def output_layer(self):
        """ convenient function for accuracy calculation """
        return self.encoder.ctc_projection

    def generate(self, src_tokens, src_lengths, blank_idx=0, **unused):
        return generate(self, src_tokens, src_lengths, blank_idx=blank_idx)

    def max_decoder_positions(self):
        """Used by sequence generator."""
        return self.encoder.max_positions()


class SpeechEncoder(S2TTransformerEncoder):
    """Transformer encoder that consists of causal attention.
    """
    def __init__(self, args, src_dict, ctc_projection):
        super().__init__(args)
        self.src_dict = src_dict
        self.ctc_projection = ctc_projection

    def forward(
        self,
        src_tokens, src_lengths, return_all_hiddens=False,
        src_txt_tokens=None,  # unused
        src_txt_lengths=None,  # unused
    ):
        return super().forward(src_tokens, src_lengths, return_all_hiddens)

    def load_state_dict(self, state_dict, strict=True):
        """
        1. ignores ctc_projection if not available
        """
        ignores = ["ctc_projection.weight", ]
        cur_state_dict = self.state_dict()

        for w in ignores:
            if (w not in state_dict) or (state_dict[w].size() != cur_state_dict[w].size()):
                logger.warning("Ignoring CTC projection weights! Make sure this is intended...")
                state_dict[w] = cur_state_dict[w]

        return super().load_state_dict(state_dict, strict=strict)


@register_model_architecture(
    "s2t_speech_encoder", "s2t_speech_encoder_s"
)
def s2t_speech_encoder_s(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True

    args.encoder_layers = getattr(args, "encoder_layers", 12)  # speech 6, text 6
    s2t_transformer_s(args)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq.models.transformer import (
    Embedding,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.speech_to_text.convtransformer import (
    ConvTransformerModel,
    ConvTransformerEncoder,
    base_architecture
)

logger = logging.getLogger(__name__)


@register_model("conv_seq2seq")
class ConvSeq2SeqModel(ConvTransformerModel):
    """
    change encoder to speech encoder + ctc projection
    """
    # @staticmethod
    # def add_args(parser):
    #     """Add model-specific arguments to the parser.
    #     """
    #     super(ConvSeq2SeqModel, ConvSeq2SeqModel).add_args(parser)
    #     parser.add_argument(
    #         "--load-pretrained-decoder-from",
    #         type=str,
    #         metavar="STR",
    #         help="model to take decoder weights from (for initialization)",
    #     )

    @classmethod
    def build_encoder(cls, args, task, embed_tokens, ctc_projection):
        sp_encoder = SpeechEncoder(args, task.source_dictionary, ctc_projection)
        if getattr(args, "load_pretrained_encoder_from", None):
            sp_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=sp_encoder, checkpoint=args.load_pretrained_encoder_from
            )
            logger.info(
                f"loaded pretrained speech encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
        return sp_encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = super().build_decoder(args, task, embed_tokens)

        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
            logger.info(
                f"loaded pretrained decoder from: "
                f"{args.load_pretrained_decoder_from}"
            )
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        encoder_embed_tokens = build_embedding(
            task.source_dictionary, args.encoder_embed_dim
        )
        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        ctc_projection = nn.Linear(
            encoder_embed_tokens.weight.shape[1],
            encoder_embed_tokens.weight.shape[0],
            bias=False,
        )
        nn.init.normal_(
            ctc_projection.weight, mean=0, std=args.encoder_embed_dim ** -0.5
        )
        encoder = cls.build_encoder(args, task, encoder_embed_tokens, ctc_projection)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    def forward_ctc_projection(self, encoder_out):
        speech_states = encoder_out["encoder_out"][0]
        # ctc projection
        logits = self.encoder.ctc_projection(
            speech_states).transpose(0, 1)

        encoder_out.update({
            "encoder_logits": [logits],
            "speech_padding_mask": encoder_out["encoder_padding_mask"]
        })
        return encoder_out

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        src_txt_tokens=None,  # unused
        src_txt_lengths=None,  # unused
    ):
        """
        """
        encoder_out = self.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths
        )
        encoder_out = self.forward_ctc_projection(encoder_out)
        logits, decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        if decoder_out is None:
            decoder_out = {}
        decoder_out.update({
            "encoder_out": encoder_out,
        })
        return logits, decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        """ temp fix for layer index error when loading check"""
        pass


class SpeechEncoder(ConvTransformerEncoder):
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
        return super().forward(src_tokens, src_lengths)

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
    "conv_seq2seq", "conv_seq2seq_s"
)
def conv_seq2seq_s(args):

    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)

    args.share_decoder_input_output_embed = True

    args.encoder_normalize_before = False
    args.decoder_normalize_before = False

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    base_architecture(args)

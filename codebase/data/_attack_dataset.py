import torchaudio
from fairseq.data.audio.audio_utils import (
    convert_waveform
)
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    # get_features_or_waveform,
)
from torchaudio.compliance.kaldi import fbank
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class SpeechToTextDatasetItem(object):
    index: int
    wav: torch.Tensor
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    speaker_id: Optional[int] = None


class AttackSpeechDataset(SpeechToTextDataset):
    def __getitem__(self, index: int) -> SpeechToTextDatasetItem:
        # source = get_features_or_waveform(
        #     self.audio_paths[index],
        #     need_waveform=self.cfg.use_audio_input,
        #     use_sample_rate=self.cfg.use_sample_rate,
        # )
        waveform, sample_rate = convert_waveform(
            *torchaudio.load(self.audio_paths[index]),
            normalize_volume=False,
            to_mono=True,
            to_sample_rate=None  # self.cfg.use_sample_rate
        )
        waveform = waveform * (2 ** 15)
        waveform.requires_grad = True  # for attack
        source = fbank(
            waveform,
            num_mel_bins=self.cfg.input_feat_per_channel,
            sample_frequency=sample_rate
        )

        if self.feature_transforms is not None:
            assert not self.cfg.use_audio_input
            source = self.feature_transforms(source)
        # source = torch.from_numpy(source).float()
        source = self.pack_frames(source)

        target = None
        if self.tgt_texts is not None:
            tokenized = self.get_tokenized_tgt_text(index)
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.cfg.prepend_tgt_lang_tag:
                lang_tag_idx = self.get_lang_tag_idx(
                    self.tgt_langs[index], self.tgt_dict
                )
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)

        speaker_id = None
        if self.speaker_to_id is not None:
            speaker_id = self.speaker_to_id[self.speakers[index]]
        return SpeechToTextDatasetItem(
            index=index, wav=waveform, source=source, target=target, speaker_id=speaker_id
        )

from fairseq.data.audio.feature_transforms import (
    AudioFeatureTransform,
    # CompositeAudioFeatureTransform,
    # register_audio_feature_transform,
)
from torchaudio.compliance.kaldi import fbank


# @register_audio_feature_transform("utterance_cmvn_torch")
class UtteranceCMVNPyTorch(AudioFeatureTransform):
    """Utterance-level CMVN (cepstral mean and variance normalization)"""

    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return UtteranceCMVNPyTorch(
            _config.get("norm_means", True),
            _config.get("norm_vars", True),
        )

    def __init__(self, norm_means=True, norm_vars=True):
        self.norm_means, self.norm_vars = norm_means, norm_vars

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(norm_means={self.norm_means}, norm_vars={self.norm_vars})"
        )

    def __call__(self, x):
        mean = x.mean(dim=0)
        square_sums = (x ** 2).sum(dim=0)

        if self.norm_means:
            x = x - mean
        if self.norm_vars:
            var = square_sums / x.shape[0] - mean ** 2
            std = var.clip(min=1e-10).sqrt()
            x = x / std
        return x


class FBankPyTorch(AudioFeatureTransform):
    """ denormalize: assumes the input is in normalized range,
    thus the transform will denormalize first by multiplying 2**15
    """
    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return UtteranceCMVNPyTorch(
            _config.get("denormalize", True),
            _config.get("num_mel_bins", 80),
            _config.get("sample_rate", 16000),
        )

    def __init__(self, denormalize=True, num_mel_bins=80, sample_rate=16000):
        self.denormalize = denormalize
        self.num_mel_bins = num_mel_bins
        self.sample_rate = sample_rate

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(denormalize={self.denormalize}, num_mel_bins={self.num_mel_bins}, sample_rate={self.sample_rate})"
        )

    def __call__(self, x):
        assert x.dim() == 2 and x.size(0) == 1
        if self.denormalize:
            x = x * (2 ** 15)
        return fbank(
            x,
            num_mel_bins=self.num_mel_bins,
            sample_frequency=self.sample_rate
        )

import logging

import torch
#import pad_sequence
from torch.nn.utils.rnn import pad_sequence
from transformers import HubertModel, Wav2Vec2FeatureExtractor

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        if ckpt is None:
            ckpt = "rinna/japanese-hubert-base"
        self.model = HubertModel.from_pretrained(ckpt)

    def get_downsample_rates(self, key: str = None) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        wavs = [wav.detach().to(device) for wav in wavs]
        input_values = pad_sequence(wavs, batch_first=True)
        output_values = self.model(input_values, output_hidden_states=True)

        return {"hidden_states": output_values.hidden_states}

import logging

import torch
#import pad_sequence
from torch.nn.utils.rnn import pad_sequence
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from ssl4speechsynthesis.model.lightning_module import DACBertLightningModule
import torchaudio

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        ckpt = "/home/wnakata/SSL4SpeechSynthesis/examples/ssl4speechsynthesis/kli4as6t/checkpoints/epoch-127.ckpt"
        cfg = torch.load(ckpt)["hyper_parameters"]['hparams']
        cfg.dac_path = '/home/wnakata/descript-audio-codec/runs/baseline_50Hz_librispeech/best/dac/weights.pth'
        self.model = DACBertLightningModule(cfg)
        self.model.load_state_dict(torch.load(ckpt)["state_dict"])

    def get_downsample_rates(self, key: str = None) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        wavs = [wav.detach().to(device) for wav in wavs]
        input_values = pad_sequence(wavs, batch_first=True)
        input_values = torchaudio.functional.resample(input_values, SAMPLE_RATE, 24000)
        self.model.feature_extractor.to(device)
        z, codes,latents = self.model.feature_extractor(input_values.unsqueeze(1))
        latents = latents.transpose(1, 2)
        output_values = self.model.model.forward(x=latents)
        return {"hidden_states": output_values.hidden_states}

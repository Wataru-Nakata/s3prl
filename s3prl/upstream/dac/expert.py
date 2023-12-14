import logging

import torch
#import pad_sequence
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel
from ssl4speechsynthesis.model.lightning_module import DACBertLightningModule
import torchaudio

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        self.model = AutoModel.from_pretrained("Wataru/necobert-base-ls", trust_remote_code=True)

    def get_downsample_rates(self, key: str = None) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        wavs = [wav.detach().to(device) for wav in wavs]
        input_values = pad_sequence(wavs, batch_first=True)
        print(device,self.model.device)
        input_values = torchaudio.functional.resample(input_values, SAMPLE_RATE, 24000)
        output = self.model.preprocessor(input_values.unsqueeze(1),sample_rate=24000)
        output_model = self.model({"x": input_values.unsqueeze(1)},sample_rate=24000)
        assert output.hidden_states[0].shape== output.shape 
        return {"hidden_states": (output)}

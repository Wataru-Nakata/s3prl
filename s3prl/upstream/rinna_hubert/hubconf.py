from .expert import UpstreamExpert as _UpstreamExpert


def hf_hubert_rinna(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)

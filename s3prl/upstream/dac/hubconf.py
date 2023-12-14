from .expert import UpstreamExpert as _UpstreamExpert


def dac(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)

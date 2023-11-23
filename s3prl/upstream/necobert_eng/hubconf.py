from .expert import UpstreamExpert as _UpstreamExpert


def necobert_eng(ckpt, *args, **kwargs):
    return _UpstreamExpert(ckpt, *args, **kwargs)

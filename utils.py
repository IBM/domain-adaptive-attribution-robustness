import os
import torch as ch


def scale_11(data):
    global_max = ch.max(ch.abs(data)) + 1e-9
    return (data/global_max).detach()


def get_dirname(_file):
    return os.path.dirname(os.path.realpath(_file))


def get_project_rootdir():
    return get_dirname(__file__)


def get_available_gpus():
    availables = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if availables in {""}:
        return tuple(["cpu"])
    return tuple(ch.device(f"cuda:{dev}") for dev in availables.split(","))


def batch_up(length, batch_size):
    return list(range(0, length + batch_size, batch_size))


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def get_multilabel_predicted_labels(logits, threshold: float = 0.0):
    return (logits > threshold).to(ch.int32)


def filter_data(data, fil):
    if isinstance(data, tuple):
        return tuple(e for i, e in enumerate(data) if fil[i])
    if isinstance(data, list):
        return [e for i, e in enumerate(data) if fil[i]]
    if isinstance(data, ch.Tensor):
        return data[fil, ...]
    if isinstance(data, dict):
        return {k: filter_data(v, fil) for k, v in data.items()}


def flatten(list_of_lists):
    return [e for l in list_of_lists for e in l]
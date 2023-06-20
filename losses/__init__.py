import torch as ch
import torch.nn.functional as f


def _cosine(ref_data, mod_data, normalize_to_loss=False):
    v_data, v_ref_data = mod_data.view(-1), ref_data.view(-1)
    _cos = f.cosine_similarity(v_data, v_ref_data, 0, 1e-8)
    if normalize_to_loss:
        return ch.ones_like(_cos) - (_cos + ch.ones_like(_cos)) / (2.0 * ch.ones_like(_cos))
    return _cos


loss_map = {"cos": _cosine}

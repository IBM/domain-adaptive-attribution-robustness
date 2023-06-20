import torch as ch
from typing import AnyStr, List, Tuple

from captum.attr import Saliency, IntegratedGradients, DeepLift  # noqa: F401


class Attention(object):
    def __init__(self, model):
        self.model = model

    def attribute(self, inputs, additional_forward_args=None, **kwargs):
        if additional_forward_args is None:
            additional_forward_args = tuple()

        _, att_weights = self.model(inputs, *additional_forward_args, return_att_weights=True)
        return att_weights.unsqueeze(-1)


explainer_map = {"ig": IntegratedGradients, "s": Saliency, "dl": DeepLift, "a": Attention}


def wordwise_embedding_attributions(attributions: ch.Tensor, word_ranges: List[Tuple[int]], word_mask: List[List[bool]], reduction: AnyStr = "mean") -> List[ch.Tensor]:

    reduced_expl = []
    for s_idx, expl_v in enumerate(attributions.unbind()):

        sample_expl = ch.zeros(size=(sum(word_mask[s_idx]),) + expl_v.size()[1:], device=attributions.device,
                               dtype=attributions.dtype)
        w_ctr = 0
        for w_idx, (w_start, w_end) in enumerate(word_ranges[s_idx]):
            if word_mask[s_idx][w_idx]:
                if reduction == "mean":
                    sample_expl[w_ctr] = ch.mean(expl_v[w_start:w_end, ...], dim=0)
                    w_ctr += 1
                else:
                    raise NotImplementedError(f"Reduction {reduction} not supported")
        if w_ctr != sum(word_mask[s_idx]):
            raise ValueError("Not all words accounted for in attribution reduction")
        reduced_expl.append(sample_expl.flatten())
    return reduced_expl


def wordwise_attributions(_input):
    return _input.sum(-1)
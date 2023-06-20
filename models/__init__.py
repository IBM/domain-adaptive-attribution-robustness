import torch as ch
from collections import OrderedDict


class BaseModel(ch.nn.Module):
    input_shape = None
    num_classes = None

    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

    def _load_weights(self, path):
        loaded_state_dict = ch.load(path, map_location="cpu")["state_dict"]
        self.load_state_dict(OrderedDict([(k, v) for k, v in loaded_state_dict.items() if k in self.state_dict()]))


from .biolinkbert import BioLinkBERT
from .roberta import RoBERTa
from .clinicallongformer import ClinicalLongformer

model_map = {"roberta": RoBERTa, "biolinkbert": BioLinkBERT, "clinicallongformer": ClinicalLongformer}

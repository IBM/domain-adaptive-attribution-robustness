import torch as ch
from collections import OrderedDict
from transformers import BertForMaskedLM
from data import data_length_map
from tokenizer_wrappers import tokenizer_map


class PubMedBERTMLM(ch.nn.Module):
    tokenizer_name = "pubmedbert"

    def __init__(self, input_shape, weight_path: str = None):
        super().__init__()
        self.input_shape = input_shape
        self.name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

        self.model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=self.name, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        self.model.eval()

        self.tokenizer = tokenizer_map[self.tokenizer_name]()
        if weight_path is not None:
            self._load_weights(weight_path)

    def forward(self, input_tokens, input_ids, token_type_ids, attention_mask, labels=None):
        self.model.eval()
        with ch.no_grad():
            preds = self.model(input_ids, attention_mask, token_type_ids, labels=labels)
            return preds

    def _load_weights(self, path):
        self.model.load_state_dict(OrderedDict([(k.replace("model.model.", ""), v) for k, v in ch.load(path)["state_dict"].items()]))

    @staticmethod
    def init_for_dataset(dataset: str, weight_path: str = None):
        return PubMedBERTMLM(input_shape=(min(512, data_length_map.get(dataset, 512)),), weight_path=weight_path)

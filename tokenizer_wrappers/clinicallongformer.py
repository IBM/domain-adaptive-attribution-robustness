import torch as ch
from transformers import LongformerTokenizerFast, AutoTokenizer, AutoModel
from tokenizer_wrappers.transformer_tokenizer import TransformerTokenizer


class ClinicalLongformerTokenizer(TransformerTokenizer):
    def __init__(self):
        super().__init__()
        self.lang = "yikuan8/Clinical-Longformer"
        self.unk_token = "<unk>"
        self.zero_token = "<unk>"
        self.sep_token = "</s>"
        self.pad_token = "<pad>"
        self.cls_token = "<s>"
        self.mask_token = "<mask>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"

        # Special tokens
        self.subtoken_prefix = None
        self.token_space_prefix = "Ġ"

        self.tok = LongformerTokenizerFast.from_pretrained(self.lang, do_lower_case=True, unk_token=self.unk_token, sep_token=self.sep_token, pad_token=self.pad_token, cls_token=self.cls_token, mask_token=self.mask_token, clean_text=True)

        self._update_special_tokens()

    def tokenize(self, batch_of_sentences, max_length, device=None):
        return_val = super().tokenize(batch_of_sentences, max_length, device)
        return_val["tokenized"]["global_attention_mask"] = ch.zeros_like(return_val["tokenized"]["attention_mask"])
        return_val["tokenized"]["global_attention_mask"][:, 0] = 1
        return return_val

    def _tokenize(self, batch_of_sentences, max_length):
        enc = super()._tokenize(batch_of_sentences, max_length)
        enc["global_attention_mask"] = ch.zeros_like(enc["attention_mask"])
        enc["global_attention_mask"][:, 0] = 1
        return enc

    def _update_tokenized(self, data_dict, max_length):
        enc = super()._update_tokenized(data_dict, max_length)
        enc["global_attention_mask"] = ch.zeros_like(enc["attention_mask"])
        enc["global_attention_mask"][:, 0] = 1
        return enc
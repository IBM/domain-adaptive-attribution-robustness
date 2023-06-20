from transformers import BertTokenizerFast
from tokenizer_wrappers.transformer_tokenizer import TransformerTokenizer


class BioLinkBERTTokenizer(TransformerTokenizer):
    def __init__(self):
        super().__init__()
        self.lang = "michiyasunaga/BioLinkBERT-base"
        self.unk_token = "[UNK]"
        self.zero_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"
        self.bos_token = None
        self.eos_token = None

        # Special tokens
        self.subtoken_prefix = "##"
        self.token_space_prefix = None

        self.tok = BertTokenizerFast.from_pretrained(self.lang, do_lower_case=True, unk_token=self.unk_token, sep_token=self.sep_token, pad_token=self.pad_token, cls_token=self.cls_token, mask_token=self.mask_token, clean_text=True)

        self._update_special_tokens()

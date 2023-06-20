from transformers import RobertaTokenizerFast
from tokenizer_wrappers.transformer_tokenizer import TransformerTokenizer


class RoBERTaTokenizer(TransformerTokenizer):
    def __init__(self):
        super(RoBERTaTokenizer, self).__init__()
        self.lang = "roberta-base"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.sep_token = "</s>"
        self.cls_token = "<s>"
        self.unk_token = "<unk>"
        self.zero_token = "<unk>"
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"

        self.subtoken_prefix = None  # No need, because if token_space_prefix not present, it's a subword
        self.token_space_prefix = "Ġ"

        self.tok = RobertaTokenizerFast.from_pretrained(self.lang, do_lower_case=True, bos_token=self.bos_token, eos_token=self.eos_token, sep_token=self.sep_token, cls_token=self.cls_token, unk_token=self.unk_token, pad_token=self.pad_token, mask_token=self.mask_token, add_prefix_space=True)

        self._update_special_tokens()

    def get_word_ranges(self, batch_tokenized_sentences: list) -> list:
        word_ranges = [[] for _ in batch_tokenized_sentences]
        for s_idx, sentence_tokens in enumerate(batch_tokenized_sentences):
            token_i = 0
            while token_i < len(sentence_tokens):
                next_token_i = token_i + 1
                while next_token_i < len(sentence_tokens):
                    if sentence_tokens[next_token_i].startswith(self.token_space_prefix):
                        break
                    if sentence_tokens[next_token_i] in self.special_tokens and sentence_tokens[token_i].replace(elf.token_space_prefix, "") != "":
                        break
                    next_token_i += 1
                word_ranges[s_idx].append((token_i, next_token_i))
                token_i = next_token_i
        return word_ranges

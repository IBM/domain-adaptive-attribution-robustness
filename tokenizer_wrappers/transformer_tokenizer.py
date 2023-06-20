import torch as ch
from tokenizer_wrappers.common import BaseTokenizer, update_input_data
from constants import WRONG_ORDS


class TransformerTokenizer(BaseTokenizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    """
    Encode methods
    """
    def _tokenize(self, batch_of_sentences, max_length):
        enc = self.tok(list(batch_of_sentences), max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        enc["input_tokens"] = list(list(self.tok.convert_ids_to_tokens(list(tok_sen))) for tok_sen in enc["input_ids"])
        # This can only create 0 as token_type_id
        enc["token_type_ids"] = ch.zeros_like(enc["attention_mask"])
        return enc

    def _update_tokenized(self, data_dict, max_length):
        batch_of_sentences = self.decode_from_words(data_dict["input_words"], data_dict["word_mask"])
        enc = self._tokenize(batch_of_sentences, max_length)
        updated_data = update_input_data(enc, seq_len=max_length, pad_token=self.pad_token, pad_id=self.tok.pad_token_id)

        return {"input_tokens": updated_data["input_tokens"], "input_ids": updated_data["input_ids"], "token_type_ids": updated_data["token_type_ids"], "attention_mask": updated_data["attention_mask"]}

    def tokenize_from_sentences(self, batch_of_sentences, max_length):
        enc = self._tokenize(batch_of_sentences, max_length)
        word_ranges = self.get_word_ranges(enc["input_tokens"])
        words, word_mask, token_word_mask = self.extract_words(enc["input_tokens"], word_ranges)

        sentences = self.decode_from_words(words, word_mask)

        # Retokenize to make sure decoding and encoding leads to same data
        num_tries = 0
        while num_tries < 3:
            enc = self._tokenize(self.decode_from_words(words, word_mask), max_length)
            word_ranges = self.get_word_ranges(enc["input_tokens"])
            words, word_mask, token_word_mask = self.extract_words(enc["input_tokens"], word_ranges)
            sentences_new = self.decode_from_words(words, word_mask)
            if sentences_new == sentences:
                break
            else:
                num_tries += 1

        data_dict = {"sentences": self.decode_from_words(words, word_mask), "word_ranges": word_ranges, "input_words": words, "word_mask": word_mask, "token_mask": token_word_mask}
        data_dict["tokenized"] = self._update_tokenized(data_dict, max_length)

        return data_dict

    """
    Decode methods
    """
    def decode_from_tokenized(self, tokenized_dict: dict, remove_special_tokens=True, remove_pad_token=True) -> list:
        to_remove = set()
        if remove_pad_token:
            to_remove.add(self.pad_token)
        if remove_special_tokens:
            to_remove.update(self.special_tokens)
        sents = list(self.tok.batch_decode([[t for t in self.tok.convert_tokens_to_ids(
            [w for w in sent if w not in to_remove])] for sent in tokenized_dict["input_tokens"]], skip_special_tokens=False, clean_up_tokenization_spaces=True))
        sents = ["".join([c for c in s if ord(c) not in WRONG_ORDS]) for s in sents]
        return sents

import torch as ch


class BaseTokenizer(object):
    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)

    def tokenize(self, batch_of_sentences, max_length, device=None):
        return_val = self.tokenize_from_sentences(batch_of_sentences, max_length)
        # Move tensors to specific device
        if device is not None:
            return_val["tokenized"]["input_ids"] = return_val["tokenized"]["input_ids"].to(device)
            return_val["tokenized"]["token_type_ids"] = return_val["tokenized"]["token_type_ids"].to(device)
            return_val["tokenized"]["attention_mask"] = return_val["tokenized"]["attention_mask"].to(device)
        return return_val

    def tokenize_from_words(self, input_words, word_mask, max_length, device=None):
        return self.tokenize(self.decode_from_words(input_words, word_mask), max_length, device=device)

    def get_word_ranges(self, batch_tokenized_sentences: list) -> list:
        word_ranges = [[] for _ in batch_tokenized_sentences]
        # """
        # No subword tokenization
        # """
        if self.subtoken_prefix is None and self.token_space_prefix is None:
            for s_idx, sentence_tokens in enumerate(batch_tokenized_sentences):
                word_ranges[s_idx] = [(i, i + 1) for i in range(len(sentence_tokens))]
        # """
        # Subword tokenization
        # """
        else:
            for s_idx, sentence_tokens in enumerate(batch_tokenized_sentences):
                token_i = 0
                while token_i < len(sentence_tokens):
                    if sentence_tokens[token_i] in self.special_tokens:
                        word_ranges[s_idx].append((token_i, token_i + 1))
                        token_i += 1
                        continue
                    if self.token_space_prefix is None and self.subtoken_prefix is not None:
                        next_token_i = token_i + 1
                        while next_token_i < len(sentence_tokens) and sentence_tokens[next_token_i].startswith(self.subtoken_prefix):
                            next_token_i += 1
                        word_ranges[s_idx].append((token_i, next_token_i))
                        token_i = next_token_i
                    elif self.subtoken_prefix is None and self.token_space_prefix is not None:
                        next_token_i = token_i + 1
                        while next_token_i < len(sentence_tokens) and not sentence_tokens[next_token_i].startswith(self.token_space_prefix) and sentence_tokens[next_token_i] not in self.special_tokens:
                            next_token_i += 1
                        word_ranges[s_idx].append((token_i, next_token_i))
                        token_i = next_token_i
                    else:
                        raise NotImplementedError("Ranges with both subtoken prefix and space prefix not implemented")
        return word_ranges

    def extract_words(self, batch_input_tokens: list, word_ranges: list) -> tuple:
        batch_words = []
        for s_idx, tokens in enumerate(batch_input_tokens):
            stripped_tokens = [tok.replace(self.subtoken_prefix, "") if self.subtoken_prefix else tok for tok in tokens]
            stripped_tokens = [tok.replace(self.token_space_prefix, "") if self.token_space_prefix else tok for tok in stripped_tokens]
            words = ["".join(stripped_tokens[word_range[0]: word_range[1]]) for word_range in word_ranges[s_idx]]
            batch_words.append(words)

        word_mask = [[word not in self.special_tokens.difference({self.mask_token, self.unk_token}) for word in words] for words in batch_words]
        token_word_mask = []
        for s_idx, word_range in enumerate(word_ranges):
            sample_token_mask = []
            for w_idx, (w_start, w_end) in enumerate(word_range):
                for t_idx in range(w_start, w_end):
                    sample_token_mask.append(word_mask[s_idx][w_idx])
            token_word_mask.append(sample_token_mask)

        return batch_words, word_mask, token_word_mask

    def _update_special_tokens(self):
        self.special_tokens = {self.unk_token, self.sep_token, self.pad_token, self.cls_token, self.mask_token, self.bos_token, self.eos_token, self.zero_token}

    @staticmethod
    def decode_from_words(batch_of_word_lists, word_mask):
        sentences = [" ".join([w for w_idx, w in enumerate(word_list) if word_mask[s_idx][w_idx]]) for s_idx, word_list in enumerate(batch_of_word_lists)]
        return sentences

    @staticmethod
    def get_valid_words(batch_of_word_lists, word_mask):
        return [[w for w_idx, w in enumerate(word_list) if word_mask[s_idx][w_idx]] for s_idx, word_list in enumerate(batch_of_word_lists)]

    def convert_to_lower(self, _str):
        _str = _str.lower()
        for _tok in self.special_tokens:
            if _tok is not None and _tok.lower() in _str:
                _str = _str.replace(_tok.lower(), _tok)
        return _str


def update_input_data(data_dict, seq_len, pad_token, pad_id=0):
    with ch.no_grad():
        for _i in range(len(data_dict["input_tokens"])):
            data_dict["input_tokens"][_i] = data_dict["input_tokens"][_i] + (seq_len - len(data_dict["input_tokens"][_i])) * [pad_token]

        helper_in_ids = ch.zeros(size=(data_dict["input_ids"].size(0), seq_len), dtype=data_dict["input_ids"].dtype, device=data_dict["input_ids"].device)
        # Add the ID of pad token
        helper_in_ids += pad_id
        # Fill data with actual IDs
        for _i, _m in enumerate(data_dict["input_ids"].unbind()):
            helper_in_ids[_i, :len(_m)] = _m
        data_dict["input_ids"] = helper_in_ids

        helper_mask = ch.zeros(size=(data_dict["attention_mask"].size(0), seq_len), dtype=data_dict["attention_mask"].dtype, device=data_dict["attention_mask"].device)
        for _i, _m in enumerate(data_dict["attention_mask"].unbind()):
            helper_mask[_i, :len(_m)] = _m
        data_dict["attention_mask"] = helper_mask

        helper_ids = ch.zeros(size=(data_dict["token_type_ids"].size(0), seq_len), dtype=data_dict["token_type_ids"].dtype, device=data_dict["token_type_ids"].device)
        for _i, _m in enumerate(data_dict["token_type_ids"].unbind()):
            helper_ids[_i, :len(_m)] = _m
        data_dict["token_type_ids"] = helper_ids

    return data_dict
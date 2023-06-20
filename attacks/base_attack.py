from copy import deepcopy

import nltk
import torch as ch

from candidate_extractors import candidate_extractor_map
from constants import STOP_WORDS


class BaseAttack(ch.nn.Module):

    # Adapted from https://huggingface.co/docs/transformers/master/en/task_summary
    def __init__(self, candidate_extractor_name: str):
        super().__init__()
        self.candidate_extractor_name = candidate_extractor_name
        self.candidate_extractor = candidate_extractor_map[candidate_extractor_name].init_for_dataset(dataset=None, weight_path=None)
        # Set parameters non-trainable
        for param in self.parameters():
            param.requires_grad = False

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)

    def get_candidate_extractor_fn(self, candidate_extractor_name):
        if candidate_extractor_name == "synonym":
            return self.get_candidates_syn
        else:
            return self.get_candidates_mlm

    def get_candidates_syn(self, input_words, word_mask, model, tokenizer, top_indices, n_candidates, orig_pos,
                           stop_word_filter=None, min_embedding_sim=0.0, device=None):
        with ch.no_grad():
            if stop_word_filter is None:
                stop_word_filter = {}

            candidates = [None for _ in range(len(input_words))]

            for s_idx, t_indices in enumerate(top_indices):

                if t_indices is not None:  # This is the batch of top indices for a given sample
                    tmp_cand = [None for _ in range(len(t_indices))]  # Initialize candidates for the batch of top indices
                    for m_idx, t_idx in enumerate(t_indices):  # Iterate over the batch of top indices for given sample
                        if t_idx is not None:  # If the given top index is not None
                            orig_word = input_words[s_idx][t_idx]

                            syns = list(self.candidate_extractor(orig_word, n=5*n_candidates, min_sim=min_embedding_sim, stop_filter=STOP_WORDS if stop_word_filter else set()))
                            syns = [(t.strip(" "), s) for (t, s) in syns if not t.startswith("##") and t != orig_word]
                            filtered_syns = []
                            n_ctr = 0
                            for syn, s in syns:
                                new_tokens = [t if _i != t_idx else syn for _i, t in enumerate(input_words[s_idx])]
                                if len(input_words[s_idx]) != len(new_tokens):
                                    raise ValueError("Token length inconsistency in POS filter")
                                n_ctr += 1
                                new_pos = self.pos_tag_batch([new_tokens])[0][t_idx]
                                if new_pos == orig_pos[s_idx][t_idx]:
                                    filtered_syns.append((syn, s))
                                if len(filtered_syns) >= n_candidates:
                                    break
                            syns = filtered_syns

                            tmp_cand[m_idx] = syns
                    candidates[s_idx] = tmp_cand
        return candidates

    def get_candidates_mlm(self, input_words, word_mask, model, tokenizer, top_indices, n_candidates, orig_pos, stop_word_filter=None, min_embedding_sim=0.0, device=None):
        with ch.no_grad():
            if stop_word_filter is None:
                stop_word_filter = {}

            masked_input_words = deepcopy(input_words)

            for s_idx, mask_indices in enumerate(top_indices):
                if mask_indices is not None:
                    for mask_idx in mask_indices:
                        if mask_idx is not None:
                            masked_input_words[s_idx][mask_idx] = self.candidate_extractor.tokenizer.mask_token

            masked_sentences = tokenizer.decode_from_words(masked_input_words, word_mask)
            masked_data = self.candidate_extractor.tokenizer(masked_sentences, max_length=model.input_shape[0],
                                                             device=device)

            # Extract mapping from top_indices to actual masked token index
            top_word_idx_to_masked_token_idx = [{ti: None for ti in s_top_indices} for s_top_indices in top_indices]
            for s_idx, s_top_indices in enumerate(top_indices):
                if s_top_indices is None:
                    continue
                tmp_masked_idx = 0
                # If index is None, nothing should be masked, therefore we can just filter those indices out
                for ti in sorted([s for s in s_top_indices if s is not None]):  # Sort to have correct ordering
                    if ti is None:
                        continue
                    if tmp_masked_idx >= len(masked_data["tokenized"]["input_tokens"][s_idx]):
                        break
                    while masked_data["tokenized"]["input_tokens"][s_idx][tmp_masked_idx] != self.candidate_extractor.tokenizer.mask_token:
                        if tmp_masked_idx >= len(masked_data["tokenized"]["input_tokens"][s_idx]) - 1:
                            break
                        tmp_masked_idx += 1
                    if masked_data["tokenized"]["input_tokens"][s_idx][tmp_masked_idx] == self.candidate_extractor.tokenizer.mask_token:
                        top_word_idx_to_masked_token_idx[s_idx][ti] = tmp_masked_idx
                        tmp_masked_idx += 1

            # Initialize candidates
            candidates = [[None] * len(top_indices[s_idx]) for s_idx in range(len(top_indices))]
            token_logits = self.candidate_extractor(**masked_data["tokenized"]).logits
            for s_idx, s_top_indices in enumerate(top_indices):
                for t_idx, ti in enumerate(s_top_indices):
                    mask_token_idx = top_word_idx_to_masked_token_idx[s_idx][ti]
                    if mask_token_idx is None:
                        continue
                    mask_token_logits = token_logits[s_idx, mask_token_idx, :]
                    mask_token_probs = ch.nn.functional.softmax(mask_token_logits, dim=0)
                    # Get tokens with larges logits, use twice the num_c to ~ get enough candidates that are not subtokens
                    top_tokens = ch.topk(mask_token_logits, k=5*n_candidates).indices.tolist()
                    top_logits = mask_token_probs[top_tokens]
                    # Decode tok tokens
                    decoded_top_tokens = [(self.candidate_extractor.tokenizer.tok.decode([t]), top_logits[_i]) for _i, t in enumerate(top_tokens)]
                    # Filter for subtokens, they are not allowed, as they result in different word length
                    if self.candidate_extractor.tokenizer.subtoken_prefix is not None:
                        decoded_top_tokens = [t for t in decoded_top_tokens if not t[0].startswith(
                            self.candidate_extractor.tokenizer.subtoken_prefix)]
                    decoded_top_tokens = [(t[0].strip(" "), t[1]) for t in decoded_top_tokens]
                    # Sometimes, it would fill in the same word, do not allow that
                    decoded_top_tokens = [t for t in decoded_top_tokens if t[0] != input_words[s_idx][ti]]
                    # Stop word filter
                    decoded_top_tokens = [t for t in decoded_top_tokens if t[0] not in stop_word_filter]
                    # Filter out special tokens
                    decoded_top_tokens = [t for t in decoded_top_tokens if t[0] not in tokenizer.special_tokens and t[0] not in
                                          self.candidate_extractor.tokenizer.special_tokens]
                    filtered_syns = []
                    for (syn, s) in decoded_top_tokens:
                        if s < min_embedding_sim:
                            continue
                        filtered_syns.append((syn, s))
                        if len(filtered_syns) >= n_candidates:
                            break
                    decoded_top_tokens = filtered_syns
                    # Add the proper number of candidates to the final list
                    candidates[s_idx][t_idx] = decoded_top_tokens[:n_candidates]
        return candidates

    @staticmethod
    def repeat(data, seq_lens):
        return [deepcopy(d) for s_idx, d in enumerate(data) for _ in range(seq_lens[s_idx])]

    @staticmethod
    def tensor_repeat(data, seq_lens):
        return ch.repeat_interleave(data, ch.tensor(seq_lens, device=data.device), dim=0)

    @staticmethod
    def pos_tag_batch(input_words):
        return [[t[1] for t in nltk.pos_tag(sent_words, tagset="universal")] for sent_words in input_words]

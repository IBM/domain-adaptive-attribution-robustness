import random
from copy import deepcopy
from typing import Collection

import torch as ch
import torch.nn.functional as f

from constants import STOP_WORDS, EPS
from explainers import wordwise_embedding_attributions, wordwise_attributions
from utils import batch_up
from .base_attack import BaseAttack


class A2T(BaseAttack):
    def __init__(self, candidate_extractor_name: str):
        super().__init__(candidate_extractor_name=candidate_extractor_name)

    def attack(self, sentences: Collection[str], labels: ch.Tensor, model, tokenizer_module: ch.nn.Module, embedding_module: ch.nn.Module,
            device: ch.device, rho_max: float = 0.15, rho_batch: float = 0.05, num_candidates: int = 5, stop_word_filter: bool = True,
            random_word_importance: bool = False, random_synonym: bool = False, min_embedding_sim: float = 0.0, multilabel: bool = False,
            pos_weight: float = None, attack_recall: bool = False):
        with ch.no_grad():
            if pos_weight is not None:
                pos_weight = pos_weight.to(labels.device)
            tmp_train = model.training
            model.eval()

            BATCH_SIZE = len(sentences)
            tokenized = tokenizer_module(sentences, max_length=model.input_shape[0], device=device)
            orig_pos = self.pos_tag_batch(tokenized["input_words"])

            adv_tokenized = deepcopy(tokenized)
            distances = [ch.zeros(size=(sum(tokenized["word_mask"][s_idx]),), dtype=ch.float32, device=device) + EPS
                         for s_idx in range(BATCH_SIZE)]
            # Get original labels
            model.zero_grad()
            embeds = embedding_module(tokenized)
            embeds["input_embeds"] = embeds["input_embeds"].detach()
        with ch.enable_grad():
            embeds["input_embeds"].requires_grad = True
            preds = model(**embeds)

            if multilabel:
                loss = f.binary_cross_entropy_with_logits(preds, labels.to(ch.float32), reduction="none", pos_weight=pos_weight).sum(-1)
                pos_logits = preds * labels
            else:
                loss = f.cross_entropy(preds, labels, weight=pos_weight, reduction="none")

            if random_word_importance:
                distances = [d + ch.rand(size=d.size(), dtype=d.dtype, device=d.device) + EPS for d in distances]
            else:
                loss.backward(inputs=embeds["input_embeds"], gradient=ch.ones_like(loss))
                grads = embeds["input_embeds"].grad.data.detach().clone()
                grads = wordwise_attributions(grads)
                grads = wordwise_embedding_attributions(grads, tokenized["word_ranges"], tokenized["word_mask"])
                for s_idx, expl in enumerate(grads):
                    if expl.size() != distances[s_idx].size():
                        raise ValueError("Explanations and distances not of equal size")
                distances = grads

        with ch.no_grad():
            argsorted_distances = [ch.argsort(d, dim=0, descending=True) for d in distances]

            distance_sorted_word_indices = [ch.zeros_like(sd) for sd in argsorted_distances]
            for s_idx, word_mask in enumerate(adv_tokenized["word_mask"]):
                v_ctr = 0
                idx_map = {}
                for w_idx, v_word in enumerate(word_mask):
                    if v_word:
                        idx_map[v_ctr] = w_idx
                        v_ctr += 1
                if v_ctr != sum(word_mask):
                    raise ValueError("Not all indices mapped to true 'input_words' index")
                for w_idx, _idx in enumerate(argsorted_distances[s_idx]):
                    distance_sorted_word_indices[s_idx][w_idx] = idx_map[_idx.item()]

            del distances, argsorted_distances

            s_lengths = [sum(s) for s in adv_tokenized["word_mask"]]
            tmp_max_l = max([len(x) for x in distance_sorted_word_indices])
            max_num_c = [max(0, int(l * rho_max)) for s_idx, l in enumerate(s_lengths)]
            s_batch_sizes = [min(max_num_c[s_idx], max(1, int(l * rho_batch))) for s_idx, l in enumerate(s_lengths)]

            candidate_index_batches = [
                [(t_indices.tolist() + [None] * (tmp_max_l - len(t_indices)))[i: i + s_batch_sizes[s_idx]]
                 for i in (range(0, len(t_indices), s_batch_sizes[s_idx]) if s_batch_sizes[s_idx] > 0 else [])]
                for s_idx, t_indices in enumerate(distance_sorted_word_indices)]
            tmp_max_l = max([len(x) for x in candidate_index_batches])
            candidate_index_batches = [c + [[None] * s_batch_sizes[s_idx]] * (tmp_max_l - len(c)) for s_idx, c in
                                       enumerate(candidate_index_batches)]

            max_dist = loss.detach().clone()  # Keep track of maximal distance for each sample
            if attack_recall:
                max_dist = - pos_logits.sum(-1).detach().clone()
            replacements = [{} for _ in range(BATCH_SIZE)]  # Create a dict of replacements for each sample

            for b_top_i in range(model.input_shape[0]):
                b_indices = [candidate_index_batches[s_idx][b_top_i] if len(candidate_index_batches[s_idx]) > b_top_i
                             else None for s_idx in range(BATCH_SIZE)]

                if all(b is None for b in b_indices):
                    continue

                if all(len(replacements[s_idx]) + 1 > sum(adv_tokenized["word_mask"][s_idx]) * rho_max
                       for s_idx in range(BATCH_SIZE)):
                    continue
                with ch.no_grad():
                    syns = self.get_candidate_extractor_fn(self.candidate_extractor_name)(
                        deepcopy(adv_tokenized["input_words"]), deepcopy(adv_tokenized["word_mask"]), model, tokenizer_module,
                        n_candidates=num_candidates, top_indices=b_indices, min_embedding_sim=min_embedding_sim, device=device,
                        orig_pos=orig_pos
                    )

                for tmp_b_idx in range(max(len(s) for s in syns if s is not None)):
                    top_indices = [b_indices[s_idx][tmp_b_idx] if len(b_indices[s_idx]) > tmp_b_idx else None
                                   for s_idx in range(BATCH_SIZE)]

                    top_indices = [None if len(replacements[s_idx]) + 1 > sum(adv_tokenized["word_mask"][s_idx])
                                   * rho_max else _idx for s_idx, _idx in enumerate(top_indices)]
                    # Get all current words
                    curr_words = [adv_tokenized["input_words"][s_idx][top_idx] if top_idx is not None else None
                                  for s_idx, top_idx in enumerate(top_indices)]

                    curr_words = [w if w not in STOP_WORDS or not stop_word_filter else None for w in curr_words]
                    # If all current words are either None or special tokens, continue
                    if all(w in {None}.union(tokenizer_module.special_tokens) for w in curr_words):
                        continue

                    curr_syns = [
                        syns[s_idx][tmp_b_idx] if syns[s_idx] is not None and len(syns[s_idx]) > tmp_b_idx and
                                                  syns[s_idx][tmp_b_idx] is not None and top_indices[s_idx]
                                                  is not None else [] for s_idx in range(BATCH_SIZE)
                    ]
                    syn_lens = [len(curr_syn) for curr_syn in curr_syns]
                    if sum(syn_lens) < 1:
                        continue

                    with ch.no_grad():
                        adv_preds = model(**embedding_module(adv_tokenized))
                        if multilabel:
                            adv_correct = ch.ones_like(max_dist, dtype=ch.bool)
                        else:
                            adv_correct = ch.argmax(adv_preds, dim=1) == labels
                    curr_syns = [syn_set if adv_correct[s_idx] else [] for s_idx, syn_set in enumerate(curr_syns)]
                    """
                    Check prediction and choose the one that maximizes cross-entropy loss
                    """
                    # If random synonym, choose a random one
                    if random_synonym:
                        curr_syns = [list(random.choices(syn_set, k=1)) if len(syn_set) > 0 else tuple() for syn_set in
                                     curr_syns]
                    syn_lens = [len(curr_syn) for curr_syn in curr_syns]
                    if sum(syn_lens) < 1:
                        continue

                    adv_words_il = self.repeat(adv_tokenized["input_words"], syn_lens)
                    adv_word_mask_il = self.repeat(adv_tokenized["word_mask"], syn_lens)
                    adv_targets_il = self.tensor_repeat(labels, syn_lens)
                    # Replace word with the synonyms
                    s_ctr = 0
                    for s_idx, syn_set in enumerate(curr_syns):
                        for _, (_syn, s) in enumerate(syn_set):
                            # Needed to make sure tokenizer tokenizes to same word
                            special_case_op = getattr(tokenizer_module, "add_token_rule", None)
                            if callable(special_case_op):
                                special_case_op(_syn, _syn)
                            adv_words_il[s_ctr][top_indices[s_idx]] = _syn
                            s_ctr += 1
                    if s_ctr != sum(syn_lens):
                        raise ValueError(f"Counter error while constructing synonym data: {s_ctr} != {sum(syn_lens)}")

                    indices = batch_up(len(adv_words_il), batch_size=BATCH_SIZE)
                    s_idx, s_ctr = 0, 0

                    for i in range(1, len(indices)):
                        tmp_words = adv_words_il[indices[i - 1]: indices[i]]
                        tmp_word_masks = adv_word_mask_il[indices[i - 1]: indices[i]]
                        tmp_targets = adv_targets_il[indices[i - 1]: indices[i]]
                        tmp_tokenized = tokenizer_module.tokenize_from_words(tmp_words, tmp_word_masks,
                                                                             max_length=model.input_shape[0],
                                                                             device=device)
                        with ch.no_grad():
                            tmp_embeds = embedding_module(tmp_tokenized)
                            tmp_preds = model(**tmp_embeds)

                        if multilabel:
                            tmp_losses = f.binary_cross_entropy_with_logits(tmp_preds, tmp_targets.to(ch.float32), reduction="none", pos_weight=pos_weight).sum(-1)

                            positive_logits = tmp_preds * tmp_targets
                            if attack_recall:
                                tmp_losses = - positive_logits.sum(-1)
                        else:
                            tmp_losses = f.cross_entropy(tmp_preds, tmp_targets, reduction="none", weight=pos_weight)

                        for tmp_idx, tmp_pred in enumerate(tmp_preds):
                            # If already all synonyms have been processed
                            while s_idx < len(syn_lens) and s_ctr >= syn_lens[s_idx]:
                                s_idx += 1
                                s_ctr = 0

                            dist = tmp_losses[tmp_idx].detach().clone()
                            if dist > max_dist[s_idx] or random_synonym:
                                max_dist[s_idx] = dist.detach().item()
                                adv_tokenized["input_words"][s_idx][top_indices[s_idx]] = deepcopy(curr_syns[s_idx][s_ctr][0])
                                replacements[s_idx][top_indices[s_idx]] = {"old": tokenized["input_words"][
                                    s_idx][top_indices[s_idx]], "new": curr_syns[s_idx][s_ctr][0], "sim": curr_syns[s_idx][s_ctr][1]}
                            s_ctr += 1

        if tmp_train:
            model.train()
        adv_sentences = tokenizer_module.decode_from_words(adv_tokenized["input_words"], adv_tokenized["word_mask"])

        return adv_sentences, replacements, max_dist

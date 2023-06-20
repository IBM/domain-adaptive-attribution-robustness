import random
from copy import deepcopy
from inspect import signature
from typing import Collection

import torch as ch
import torch.nn.functional as f

from constants import STOP_WORDS, EPS
from explainers import explainer_map
from losses import loss_map
from utils import batch_up, get_multilabel_predicted_labels
from .base_attack import BaseAttack


class ExplanationFooler(BaseAttack):

    # Adapted from https://huggingface.co/docs/transformers/master/en/task_summary
    def __init__(self, candidate_extractor_name: str):
        super().__init__(candidate_extractor_name=candidate_extractor_name)

    def attack(
            self, sentences: Collection[str], model, tokenizer_module: ch.nn.Module, embedding_module: ch.nn.Module,
            explainer_module: ch.nn.Module, device: ch.device, rho_max: float = 0.15, rho_batch: float = 0.05,
            num_candidates: int = 5, stop_word_filter: bool = True, random_word_importance: bool = False,
            random_synonym: bool = False, min_embedding_sim: float = 0.0, attribution_loss_fn_name: str = "pc",
            attribution_loss_fn_args: dict = None, gradient_word_importance: bool = True, multilabel: bool = False,
            adversarial_preds: bool = False, delta: float = 1.0, sentence_similarity_module: ch.nn.Module = None,
            min_sentence_sim: float = 0.0, pos_weight=None, detach: bool = False,
    ):
        if attribution_loss_fn_args is None:
            attribution_loss_fn_args = {"normalize_to_loss": True}
        attribution_loss_fn = loss_map[attribution_loss_fn_name]

        tmp_train = model.training
        model.eval()

        BATCH_SIZE = len(sentences)
        # Setup attack and data gathering
        tokenized = tokenizer_module(sentences, max_length=model.input_shape[0], device=device)
        embeds = embedding_module(tokenized)

        with ch.no_grad():
            if multilabel:
                labels = get_multilabel_predicted_labels(model(**embeds))
            else:
                labels = ch.argmax(model(**embeds), dim=1)

        with ch.enable_grad():
            orig_explanations = self.get_explanations(sentences, model, tokenizer_module, embedding_module, explainer_module, labels, multilabel=multilabel, device=device, detach=detach)

        with ch.no_grad():
            orig_pos = self.pos_tag_batch(tokenized["input_words"])
            adv_tokenized = deepcopy(tokenized)
            """
            Check most important words by setting them to unk_token
            """
            distances = [ch.zeros(size=(sum(tokenized["word_mask"][s_idx]),), dtype=ch.float32, device=device) + EPS for s_idx in range(BATCH_SIZE)]

            if random_word_importance:
                # Fill up tensors with random distances in [0,1)
                distances = [d + ch.rand(size=d.size(), dtype=d.dtype, device=d.device) + EPS for d in distances]
            else:
                if gradient_word_importance:
                    with ch.enable_grad():
                        wi_explainer_module = explainer_map["s"](model)
                        expls = self.get_explanations(sentences, model, tokenizer_module, embedding_module, wi_explainer_module, labels, multilabel=multilabel,
                                                      device=device, detach=detach)
                    for s_idx, expl in enumerate(expls):
                        if expl.size() != distances[s_idx].size():
                            raise ValueError("Explanations and distances not of equal size")
                    distances = expls
                    del wi_explainer_module
                else:
                    # TODO not adjusted to multilabel
                    # Create list of input words as many times as there are words in each sentence
                    adv_words_il = self.repeat(adv_tokenized["input_words"], [sum(w) for w in adv_tokenized["word_mask"]])
                    adv_word_mask_il = self.repeat(adv_tokenized["word_mask"], [sum(w) for w in adv_tokenized["word_mask"]])

                    # Replace each word once with unk_token
                    int_idx = 0
                    for s_idx, num_words in enumerate([sum(w) for w in adv_tokenized["word_mask"]]):
                        num_replaced = 0
                        for w_idx, _word_mask in enumerate(adv_tokenized["word_mask"][s_idx]):
                            if _word_mask:
                                adv_words_il[int_idx][w_idx] = tokenizer_module.unk_token
                                num_replaced += 1
                                int_idx += 1
                        if num_replaced != num_words:
                            raise ValueError("Not all valid words have been replaced by unk_token")
                    if int_idx != sum([sum(w) for w in adv_tokenized["word_mask"]]):
                        raise ValueError("Word importance ranking error with sequence lengths")

                    # Process temporary explanations in batches
                    indices = batch_up(len(adv_words_il), batch_size=BATCH_SIZE)
                    s_idx, w_ctr = 0, 0
                    for i in range(1, len(indices)):
                        tmp_words = adv_words_il[indices[i - 1]: indices[i]]
                        tmp_word_masks = adv_word_mask_il[indices[i - 1]: indices[i]]

                        tmp_data = tokenizer_module.tokenize_from_words(tmp_words, tmp_word_masks, model.input_shape[0],
                                                                        device=device)
                        tmp_embeds = embedding_module(tmp_data)
                        tmp_labels = ch.argmax(model(**tmp_embeds), dim=1).detach()
                        tmp_explanations = self.get_explanations(tokenizer_module.decode_from_words(tmp_words, tmp_word_masks),
                                                                 model, tokenizer_module, embedding_module, explainer_module,
                                                                 tmp_labels, device=device, multilabel=multilabel, detach=detach)
                        for tmp_idx, tmp_expl in enumerate(tmp_explanations):
                            s_offset = 0  # Calculate the number of non-word tokens in the beginning of the sentence
                            while not tmp_word_masks[tmp_idx][s_offset]:
                                s_offset += 1

                            if orig_explanations[s_idx].size() != tmp_expl.size():
                                dist = EPS
                            # If word is a stop word, set distance to zero to deprioritize
                            elif stop_word_filter and adv_tokenized["input_words"][s_idx][w_ctr + s_offset] in STOP_WORDS:
                                dist = EPS
                            else:
                                dist = attribution_loss_fn(ref_data=orig_explanations[s_idx], mod_data=tmp_expl, **attribution_loss_fn_args).detach()

                            distances[s_idx][w_ctr] = dist
                            w_ctr += 1
                            if w_ctr >= sum(tmp_word_masks[tmp_idx]):
                                s_idx += 1
                                w_ctr = 0
                    if w_ctr != 0:
                        raise ValueError("Word counter did not reset to zero.")
                    if s_idx != BATCH_SIZE:
                        raise ValueError("Not all samples are processed.")

                    # Delete data that is not needed anymore
                    del adv_words_il, adv_word_mask_il, tmp_words, tmp_word_masks, tmp_explanations, tmp_data

            # Argsort the distances. They need to be adjusted to reflect the actual words in "input_words"
            # Depending on front/back padding and special tokens
            argsorted_distances = [ch.argsort(d, dim=0, descending=True) for d in distances]

            # This corresponds to the converted indices in the "input_words" field
            distance_sorted_word_indices = [ch.zeros_like(sd) for sd in argsorted_distances]
            for s_idx, word_mask in enumerate(adv_tokenized["word_mask"]):
                v_ctr, idx_map = 0, {}
                for w_idx, v_word in enumerate(word_mask):
                    if v_word:
                        idx_map[v_ctr] = w_idx
                        v_ctr += 1
                if v_ctr != sum(word_mask):
                    raise ValueError("Not all indices mapped to true 'input_words' index")
                for w_idx, _idx in enumerate(argsorted_distances[s_idx]):
                    distance_sorted_word_indices[s_idx][w_idx] = idx_map[_idx.item()]

            del distances, argsorted_distances

            # Batch up top indices
            s_lengths = [sum(s) for s in adv_tokenized["word_mask"]]
            tmp_max_l = max([len(x) for x in distance_sorted_word_indices])
            max_num_c = [max(0, int(l * rho_max)) for s_idx, l in enumerate(s_lengths)]
            s_batch_sizes = [min(max_num_c[s_idx], max(1, int(l * rho_batch))) for s_idx, l in enumerate(s_lengths)]

            candidate_index_batches = [
                [(t_indices.tolist() + [None] * (tmp_max_l - len(t_indices)))[i: i + s_batch_sizes[s_idx]]
                 for i in (range(0, len(t_indices), s_batch_sizes[s_idx]) if s_batch_sizes[s_idx] > 0 else [])]
                for s_idx, t_indices in enumerate(distance_sorted_word_indices)]
            tmp_max_l = max([len(x) for x in candidate_index_batches])
            candidate_index_batches = [c + [[None] * s_batch_sizes[s_idx]] * (tmp_max_l - len(c)) for s_idx, c in enumerate(candidate_index_batches)]

            """
            Start replacing words with their synonyms
            """
            max_dist = [0.0 for _ in range(BATCH_SIZE)]  # Keep track of maximal distance for each sample
            replacements = [{} for _ in range(BATCH_SIZE)]  # Create a dict of replacements for each sample

            for b_top_i in range(model.input_shape[0]):
                b_indices = [candidate_index_batches[s_idx][b_top_i] if len(candidate_index_batches[s_idx]) > b_top_i else None for s_idx in range(BATCH_SIZE)]

                if all(b is None for b in b_indices) or all(len(replacements[s_idx]) + 1 > sum(adv_tokenized["word_mask"][s_idx]) * rho_max for s_idx in range(BATCH_SIZE)):
                    continue

                # Extract candidates for the masked words
                syns = self.get_candidate_extractor_fn(self.candidate_extractor_name)(
                    adv_tokenized["input_words"], adv_tokenized["word_mask"], model, tokenizer_module,
                    n_candidates=num_candidates, top_indices=b_indices, min_embedding_sim=min_embedding_sim, device=device, orig_pos=orig_pos)

                for tmp_b_idx in range(max(len(s) for s in syns if s is not None)):

                    # Get current top index, if number of words is large enough in sample, else None
                    top_indices = [b_indices[s_idx][tmp_b_idx] if len(b_indices[s_idx]) > tmp_b_idx else None for s_idx in range(BATCH_SIZE)]

                    # Set current idx to None if another change would result in too high perturbation ratio
                    top_indices = [None if len(replacements[s_idx]) + 1 > sum(adv_tokenized["word_mask"][s_idx]) * rho_max else _idx for s_idx, _idx in enumerate(top_indices)]
                    # Get all current words
                    curr_words = [adv_tokenized["input_words"][s_idx][top_idx] if top_idx is not None else None for s_idx, top_idx in enumerate(top_indices)]

                    curr_words = [w if w not in STOP_WORDS or not stop_word_filter else None for w in curr_words]
                    # If all current words are either None or special tokens, continue
                    if all(w in {None}.union(tokenizer_module.special_tokens) for w in curr_words):
                        continue

                    curr_syns = [syns[s_idx][tmp_b_idx] if syns[s_idx] is not None and len(syns[s_idx]) > tmp_b_idx and syns[s_idx][tmp_b_idx]
                                 is not None and top_indices[s_idx] is not None else [] for s_idx in range(BATCH_SIZE)]
                    syn_lens = [len(curr_syn) for curr_syn in curr_syns]

                    if sum(syn_lens) < 1:
                        continue

                    adv_words_il = self.repeat(adv_tokenized["input_words"], syn_lens)
                    adv_word_mask_il = self.repeat(adv_tokenized["word_mask"], syn_lens)
                    s_ctr = 0
                    for s_idx, syn_set in enumerate(curr_syns):
                        for _, (_syn, s) in enumerate(syn_set):
                            adv_words_il[s_ctr][top_indices[s_idx]] = _syn
                            s_ctr += 1
                    if s_ctr != sum(syn_lens):
                        raise ValueError(f"Counter error while constructing synonym data INF: {s_ctr} != {sum(syn_lens)}")

                    adv_tokenized_il = tokenizer_module.tokenize_from_words(adv_words_il, adv_word_mask_il, model.input_shape[0], device=device)
                    with ch.no_grad():
                        adv_embeds_il = embedding_module(adv_tokenized_il)
                        if multilabel:
                            il_logits = model(**adv_embeds_il)

                            il_c = get_multilabel_predicted_labels(il_logits) == self.tensor_repeat(labels, syn_lens)
                            il_correct = ch.eq(il_c.sum(dim=1), ch.zeros_like(il_c.sum(dim=1)) + model.num_classes)
                        else:
                            il_output = model(**adv_embeds_il)
                            il_correct = ch.argmax(il_output, dim=1) == self.tensor_repeat(labels, syn_lens)
                        if adversarial_preds:
                            il_correct = ch.ones_like(il_correct)

                    ctr = 0
                    for s_idx, syn_set in enumerate(list(curr_syns)):
                        filtered_syns = []
                        for _syn in syn_set:
                            if il_correct[ctr]:
                                filtered_syns.append(_syn)
                            ctr += 1
                        curr_syns[s_idx] = filtered_syns
                    if ctr != sum(syn_lens):
                        raise ValueError(f"Counter error while constructing synonym data: {ctr} != {sum(syn_lens)}")
                    """
                    Explain and chose the best synonym for each s_id
                    """
                    if random_synonym:
                        curr_syns = [list(random.choices(syn_set, k=1)) if len(syn_set) > 0 else tuple() for syn_set in curr_syns]

                    syn_lens = [len(curr_syn) for curr_syn in curr_syns]
                    # If no synonyms anymore, continue
                    if sum(syn_lens) < 1:
                        continue
                    # Repeat the data again with valid synonyms
                    adv_words_il = self.repeat(adv_tokenized["input_words"], syn_lens)
                    adv_word_mask_il = self.repeat(adv_tokenized["word_mask"], syn_lens)
                    adv_labels_il = self.tensor_repeat(labels, syn_lens)
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

                    # Process temporary explanations in batches
                    indices = batch_up(len(adv_words_il), batch_size=BATCH_SIZE)
                    s_idx, s_ctr = 0, 0
                    for i in range(1, len(indices)):
                        tmp_words = adv_words_il[indices[i - 1]: indices[i]]
                        tmp_word_masks = adv_word_mask_il[indices[i - 1]: indices[i]]
                        tmp_labels = adv_labels_il[indices[i - 1]: indices[i]].detach()
                        if adversarial_preds:
                            tmp_preds = model(**embedding_module(tokenizer_module( tokenizer_module.decode_from_words(tmp_words, tmp_word_masks),
                                max_length=model.input_shape[0], device=device)))
                            if multilabel:
                                tmp_loss = f.binary_cross_entropy_with_logits(tmp_preds, tmp_labels.to(ch.float32), reduction="none", pos_weight=pos_weight).sum(-1).detach()
                            else:
                                tmp_loss = f.cross_entropy(tmp_preds, tmp_labels, reduction="none", weight=pos_weight).detach()
                        with ch.enable_grad():
                            tmp_explanations = self.get_explanations(tokenizer_module.decode_from_words(tmp_words, tmp_word_masks), model, tokenizer_module, embedding_module, explainer_module,
                                tmp_labels, device=device, multilabel=multilabel, detach=detach)

                        for tmp_idx, tmp_expl in enumerate(tmp_explanations):
                            # If already all synonyms have been processed
                            while s_idx < len(syn_lens) and s_ctr >= syn_lens[s_idx]:
                                s_idx += 1
                                s_ctr = 0
                            if orig_explanations[s_idx].size() != tmp_expl.size():
                                s_ctr += 1
                                continue
                            e_dist = attribution_loss_fn(ref_data=orig_explanations[s_idx], mod_data=tmp_expl, **attribution_loss_fn_args).detach()
                            if adversarial_preds:
                                dist = delta * e_dist + tmp_loss[tmp_idx]
                            else:
                                dist = e_dist
                            if dist > max_dist[s_idx] or random_synonym:
                                max_dist[s_idx] = dist.detach().item()
                                adv_tokenized["input_words"][s_idx] = deepcopy(tmp_words[tmp_idx])
                                replacements[s_idx][top_indices[s_idx]] = {"old": tokenized["input_words"][s_idx][top_indices[s_idx]], "new": curr_syns[s_idx][s_ctr][0], "sim": curr_syns[s_idx][s_ctr][1]}
                            s_ctr += 1

        if tmp_train:
            model.train()

        adv_sentences = tokenizer_module.decode_from_words(adv_tokenized["input_words"], adv_tokenized["word_mask"])
        return adv_sentences, replacements, max_dist

    @staticmethod
    def get_explanations(sentences, model, tokenizer_module, embedding_module, explainer_module, labels: ch.Tensor = None, device: ch.device = ch.device("cpu"),
                         multilabel: bool = False, detach: bool = False) -> ch.Tensor:

        from explainers import wordwise_embedding_attributions, wordwise_attributions

        model.zero_grad()
        with ch.no_grad():
            tokenized = tokenizer_module(sentences, max_length=model.input_shape[0], device=device)
            embeds = embedding_module(tokenized)

            if labels is None:
                if multilabel:
                    labels = get_multilabel_predicted_labels(model(**embeds))
                else:
                    labels = ch.argmax(model(**embeds), dim=1)
            tmp_cudnn_be = ch.backends.cudnn.enabled
            ch.backends.cudnn.enabled = False

            attribute_kwargs = {}
            if "baselines" in signature(explainer_module.attribute).parameters:
                attribute_kwargs["baselines"] = ch.zeros_like(embeds["input_embeds"], dtype=ch.float32, device=device)
            if "n_steps" in signature(explainer_module.attribute).parameters:
                attribute_kwargs["n_steps"] = 5
            if "sliding_window_shapes" in signature(explainer_module.attribute).parameters:
                attribute_kwargs["sliding_window_shapes"] = (1,) + embeds["input_embeds"].size()[2:]
            if "internal_batch_size" in signature(explainer_module.attribute).parameters:
                attribute_kwargs["internal_batch_size"] = embeds["input_embeds"].size(0)
            additional_forward_args = tuple()
            if "attention_mask" in signature(model.forward).parameters:
                additional_forward_args += (embeds["attention_mask"],)
            if "token_type_ids" in signature(model.forward).parameters:
                additional_forward_args += (embeds["token_type_ids"],)
            if "global_attention_mask" in signature(model.forward).parameters:
                additional_forward_args += (embeds["global_attention_mask"],)
            model.zero_grad()
            ch.use_deterministic_algorithms(True, warn_only=True)

        with ch.enable_grad():
            if not multilabel:
                attrs = explainer_module.attribute(inputs=embeds["input_embeds"], target=labels, additional_forward_args=additional_forward_args, **attribute_kwargs)
            elif isinstance(explainer_module, explainer_map["a"]):
                attrs = explainer_module.attribute(inputs=embeds["input_embeds"], target=None, additional_forward_args=additional_forward_args, **attribute_kwargs)
            else:
                attrs = ch.zeros_like(embeds["input_embeds"])
                for c_idx in range(labels.size(1)):
                    nonzero_s_indices = ch.nonzero(labels[:, c_idx]).squeeze(-1)
                    if len(nonzero_s_indices) < 1:
                        continue
                    if "baselines" in signature(explainer_module.attribute).parameters:
                        attribute_kwargs["baselines"] = ch.zeros_like(embeds["input_embeds"][nonzero_s_indices, ...], dtype=ch.float32, device=device)
                    tmp_additional_args = tuple(arg[nonzero_s_indices, ...] for arg in additional_forward_args)
                    tmp_attrs = explainer_module.attribute(
                        inputs=embeds["input_embeds"][nonzero_s_indices, ...], target=ch.zeros_like(nonzero_s_indices) + c_idx, additional_forward_args=tmp_additional_args, **attribute_kwargs)
                    attrs[nonzero_s_indices] = attrs[nonzero_s_indices] + tmp_attrs.to(attrs.dtype)
                    if detach:
                        attrs = attrs.detach()

            if detach:
                attrs = attrs.detach()

            ch.use_deterministic_algorithms(True, warn_only=False)
            # Reduce last dimension to get one scalar for each word
            attrs = wordwise_attributions(attrs)
            # Here, add word ranges and process explanation
            attrs = wordwise_embedding_attributions(attrs, tokenized["word_ranges"], tokenized["word_mask"])
            ch.backends.cudnn.enabled = tmp_cudnn_be

        return attrs

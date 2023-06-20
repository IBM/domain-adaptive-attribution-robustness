import os
import argparse
import json
import numpy as np
from time import time
import torch as ch
import torch.nn.functional as f
import pytorch_lightning as pl
import torchmetrics.functional as tmf

from scripts.base_trainer import BaseClassifier, get_trainer
from models import model_map
from data import data_map
from attacks import attack_map
from explainers import explainer_map
from losses import loss_map
from sentence_encoders import sentence_encoder_map
from constants import WRONG_ORDS
from utils import filter_data, get_multilabel_predicted_labels, get_dirname


class FARClassifier(BaseClassifier):
    def __init__(self, model, explainer, datamodule, optimizer_name: str = "adam", learning_rate: float = 0.00001, weight_decay: float = 0.0,
            delta: float = 5.0, attack_name: str = "ef", candidate_extractor_name: str = "distilroberta", attack_ratio: float = 0.75, rho_max: float = 0.15,
            rho_batch: float = 0.0, num_candidates: int = 15, adversarial_preds: bool = False, relax_multilabel_constraint: bool = False,
            stop_word_filter: bool = True, random_word_importance: bool = False, random_synonym: bool = False, attribution_loss_name: str = "cos"):
        super().__init__(model=model, optimizer_name=optimizer_name, learning_rate=learning_rate, weight_decay=weight_decay, datamodule=datamodule)

        self.explainer = explainer
        self.attack_algo = attack_map[attack_name](candidate_extractor_name=candidate_extractor_name)
        self.dataset_name = datamodule.name
        self.attack_ratio = attack_ratio
        self.delta = delta
        self.rho_max = rho_max
        self.rho_batch = rho_batch
        self.num_candidates = num_candidates
        self.adversarial_preds = adversarial_preds
        self.relax_multilabel_constraint = relax_multilabel_constraint
        self.stop_word_filter = stop_word_filter
        self.random_word_importance = random_word_importance
        self.random_synonym = random_synonym
        self.attribution_loss_name = attribution_loss_name

        self.med_bce = sentence_encoder_map["med_bce"]()

    def training_step(self, batch, batch_idx):
        with ch.no_grad():
            data_sentences, labels = batch
            tokenized = self.tokenizer(data_sentences, max_length=self.model.input_shape[0], device=self.device)
            sentences = self.tokenizer.decode_from_words(tokenized["input_words"], tokenized["word_mask"])
            sentences = ["".join([c for c in s if ord(c) not in WRONG_ORDS]) for s in sentences]

        attack_start = time()
        is_attacked = np.random.random_sample(size=(len(sentences),)) < self.attack_ratio

        if sum(is_attacked) > 0:
            adv_sentences, _, _ = self.attack_algo.attack(sentences=[s for i, s in enumerate(sentences) if is_attacked[i]], model=self.model,
                multilabel=self.datamodule.multilabel, tokenizer_module=self.tokenizer, embedding_module=self.embedding, explainer_module=self.explainer,
                device=self.device, rho_max=self.rho_max, rho_batch=self.rho_batch, num_candidates=self.num_candidates, adversarial_preds=self.adversarial_preds,
                delta=self.delta, stop_word_filter=self.stop_word_filter, random_word_importance=self.random_word_importance, random_synonym=self.random_synonym,
                attribution_loss_fn_name=self.attribution_loss_name, attribution_loss_fn_args=None)
        else:
            adv_sentences = []
        self.model.zero_grad()

        attack_time = time() - attack_start

        # Overwrite adversarial sentences
        train_sentences = [adv_sentences.pop(0) if is_attacked[i] else s for i, s in enumerate(sentences)]
        if len(adv_sentences) > 0:
            raise ValueError("Not all adversarial sentences overwritten")
        with ch.no_grad():
            tokenized = self.tokenizer(train_sentences, max_length=self.model.input_shape[0], device=self.device)
            embeds = self.embedding(tokenized)
        preds = self.model(**embeds)

        orig_explanations = self.attack_algo.get_explanations(sentences, self.model, self.tokenizer, self.embedding, self.explainer, labels=labels, device=self.device, multilabel=self.multilabel)
        adv_explanations = self.attack_algo.get_explanations(train_sentences, self.model, self.tokenizer, self.embedding, self.explainer, labels=labels, device=self.device, multilabel=self.multilabel)

        if not self.multilabel:
            if self.datamodule.pos_weights is not None:
                p_loss = f.cross_entropy(preds, labels, weight=self.datamodule.pos_weights.to(preds.device))
            else:
                p_loss = f.cross_entropy(preds, labels)
        else:
            if self.datamodule.pos_weights is not None:
                p_loss = f.binary_cross_entropy_with_logits(preds, labels.to(ch.float32), pos_weight=self.datamodule.pos_weights.to(preds.device))
            else:
                p_loss = f.binary_cross_entropy_with_logits(preds, labels.to(ch.float32))

        for s_idx, (oe, ae) in enumerate(zip(orig_explanations, adv_explanations)):
            if oe.size() != ae.size():
                print("WARNING wrong explanation sizes")
        e_loss = ch.stack([loss_map[self.attribution_loss_name](oe, ae, normalize_to_loss=True) for oe, ae in zip(orig_explanations, adv_explanations) if oe.size() == ae.size()]).mean()

        loss = p_loss + self.delta * e_loss

        adv_preds, adv_labels = preds[is_attacked], labels[is_attacked]
        clean_preds, clean_labels = preds[~is_attacked], labels[~is_attacked]

        log_dict = {"a_time": float(attack_time / max(sum(is_attacked), 1)), "num_a": float(sum(is_attacked)), "num_c": float(sum(~is_attacked)),
            "loss_tr": loss, "e_loss_tr": e_loss, "p_loss_tr": p_loss, "acc_tr_a": tmf.accuracy(adv_preds, adv_labels.to(ch.int32), num_classes=self.model.num_classes) if len(adv_preds) > 0 else 0.0,
            "acc_tr_c": tmf.accuracy(clean_preds, clean_labels.to(ch.int32), num_classes=self.model.num_classes) if len(clean_preds) > 0 else 0.0,
            "f1_tr_a": tmf.f1_score(adv_preds, adv_labels.to(ch.int32), num_classes=self.model.num_classes, average="macro") if len(adv_preds) > 0 else 0.0,
            "f1_tr_c": tmf.f1_score(clean_preds, clean_labels.to(ch.int32), num_classes=self.model.num_classes, average="macro") if len(clean_preds) > 0 else 0.0}

        if self.multilabel:
            for top_k_r in [0.1, 0.16, 0.3]:
                top_k = int(round(top_k_r * self.model.num_classes))
                if top_k <= self.model.num_classes:
                    log_dict[f"p_{top_k}_tr_a"] = tmf.precision(adv_preds, adv_labels.to(ch.int32), num_classes=self.model.num_classes, top_k=top_k) if len(adv_preds) > 0 else 0.0
                    log_dict[f"p_{top_k}_tr_c"] = tmf.precision(clean_preds, clean_labels.to(ch.int32), num_classes=self.model.num_classes, top_k=top_k) if len(clean_preds) > 0 else 0.0

        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=False, batch_size=len(data_sentences))

        return loss

    def validation_step(self, batch, batch_idx):
        with ch.no_grad():
            data_sentences, labels = batch
            data_sentences = ["".join([c for c in s if ord(c) not in WRONG_ORDS]) for s in data_sentences]
            tokenized = self.tokenizer(data_sentences, max_length=self.model.input_shape[0], device=self.device)
            sentences = self.tokenizer.decode_from_words(tokenized["input_words"], tokenized["word_mask"])
            sentences = ["".join([c for c in s if ord(c) not in WRONG_ORDS]) for s in sentences]

            tokenized = self.tokenizer(sentences, max_length=self.model.input_shape[0], device=self.device)
            embeds = self.embedding(tokenized)
            preds = self.model(**embeds)

            # Filter correct preds
            if self.multilabel:
                fil = get_multilabel_predicted_labels(preds) == labels
                _d_filter = ch.eq(fil.sum(dim=1), ch.zeros_like(fil.sum(dim=1)) + self.model.num_classes)
                # Filter no labels
                _d_filter = _d_filter * (labels.sum(dim=1) > 0)
                if self.relax_multilabel_constraint:
                    _d_filter = ch.ones_like(_d_filter, dtype=ch.bool)

            else:
                _d_filter = ch.argmax(preds, dim=1) == labels

            # If no data is correctly classified
            if _d_filter.sum() < 1:
                print("No correctly predicted samples in batch")
                return (None,)*12

            sentences, labels = filter_data(sentences, fil=_d_filter), filter_data(labels, fil=_d_filter)

            from time import time
            start = time()
            adv_sentences, replacements, _ = self.attack_algo.attack(sentences=sentences, model=self.model, multilabel=self.datamodule.multilabel, tokenizer_module=self.tokenizer,
                embedding_module=self.embedding, explainer_module=self.explainer, device=self.device, rho_max=self.rho_max,
                rho_batch=self.rho_batch, num_candidates=self.num_candidates, adversarial_preds=False, delta=self.delta,
                stop_word_filter=self.stop_word_filter, random_word_importance=self.random_word_importance, random_synonym=self.random_synonym,
                attribution_loss_fn_name=self.attribution_loss_name, attribution_loss_fn_args=None, detach=True)
        self.model.zero_grad()
        avg_attack_time = (time() - start)/len(adv_sentences)

        orig_tokenized = self.tokenizer(sentences, max_length=self.model.input_shape[0], device=self.device)
        adv_tokenized = self.tokenizer(adv_sentences, max_length=self.model.input_shape[0], device=self.device)

        orig_embeds = self.embedding(orig_tokenized)
        adv_embeds = self.embedding(adv_tokenized)

        with ch.no_grad():
            if self.multilabel:
                orig_preds = self.model(**orig_embeds)
                orig_labels = get_multilabel_predicted_labels(orig_preds)
                adv_preds = self.model(**adv_embeds)
                adv_labels = get_multilabel_predicted_labels(adv_preds)
            else:
                orig_preds = f.softmax(self.model(**orig_embeds), dim=1)
                adv_preds = f.softmax(self.model(**adv_embeds), dim=1)
                orig_labels = ch.argmax(orig_preds, dim=1)
                adv_labels = ch.argmax(adv_preds, dim=1)

        orig_explanations = self.attack_algo.get_explanations(self.tokenizer.decode_from_words(orig_tokenized["input_words"], orig_tokenized["word_mask"]),
            self.model, self.tokenizer, self.embedding, self.explainer, labels=orig_labels, device=self.device,
            multilabel=self.multilabel, detach=True)
        self.model.zero_grad()
        adv_explanations = self.attack_algo.get_explanations(self.tokenizer.decode_from_words(adv_tokenized["input_words"], adv_tokenized["word_mask"]),
            self.model, self.tokenizer, self.embedding, self.explainer, labels=adv_labels, device=self.device,
            multilabel=self.multilabel, detach=True)

        e_sim = [loss_map[self.attribution_loss_name](ox, ax).detach().item() for ox, ax in zip(orig_explanations, adv_explanations) if ox.size() == ax.size()]
        e_loss = ch.stack([loss_map[self.attribution_loss_name](oe, ae, normalize_to_loss=True) for oe, ae in zip(orig_explanations, adv_explanations) if oe.size() == ae.size()])


        ss_use, ss_tse, ss_ce, ss_pp, ss_med_bce, ss_bertscore = [], [], [], [], [], []
        for s_idx, s in enumerate(sentences):
            for enc_name, enc, res_l in [("med_bce", self.med_bce, ss_med_bce)]:
                sim = enc.semantic_sim(sentence1=sentences[s_idx], sentence2=adv_sentences[s_idx], reduction="min").detach().item()
                res_l.append(sim)

        med_ks = [(el / ((1.0 - medss) + 0.0000001)).item() for el, medss in zip(e_loss, ss_med_bce)]

        return (orig_preds.detach(), adv_preds.detach(), labels.to(ch.int32).detach(), e_loss.detach(), len(e_sim), avg_attack_time, e_sim, replacements, ss_med_bce, med_ks)

    def validation_epoch_end(self, validation_step_outputs, stage_suffix="v"):
        if all([all(e is None for e in t) for t in validation_step_outputs]):
            self.log_dict({"e_loss_v_a": 1e6}, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
            return

        preds = ch.cat([t[0] for t in validation_step_outputs if t[0] is not None], dim=0)
        adv_preds = ch.cat([t[1] for t in validation_step_outputs if t[1] is not None], dim=0)
        labels = ch.cat([t[2] for t in validation_step_outputs if t[2] is not None], dim=0)
        e_loss = ch.cat([t[3] for t in validation_step_outputs if t[3] is not None], dim=0)

        if self.trainer.world_size == 1 or True:
            all_e_loss = e_loss
            all_preds = preds
            all_adv_preds = adv_preds
            all_labels = labels
        else:
            all_e_loss = ch.cat(self.all_gather(e_loss).unbind(), dim=0)
            all_preds = ch.cat(self.all_gather(preds).unbind(), dim=0)
            all_adv_preds = ch.cat(self.all_gather(adv_preds).unbind(), dim=0)
            all_labels = ch.cat(self.all_gather(labels).unbind(), dim=0)

        if not self.multilabel:
            clean_loss = f.cross_entropy(all_preds, all_labels.to(ch.int64))
            adv_loss = f.cross_entropy(all_adv_preds, all_labels.to(ch.int64))
        else:
            clean_loss = f.binary_cross_entropy_with_logits(all_preds, all_labels.to(ch.float32))
            adv_loss = f.binary_cross_entropy_with_logits(all_adv_preds, all_labels.to(ch.float32))

        log_dict = {f"loss_{stage_suffix}_c": clean_loss, f"loss_{stage_suffix}_a": adv_loss, f"avg_e_loss_{stage_suffix}_a": all_e_loss.mean()}

        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)


if __name__ == "__main__":
    with open(os.path.join(get_dirname(__file__), "far_config.json")) as json_file:
        config = json.load(json_file)
    args = argparse.Namespace(**config)

    pl.seed_everything(args.overall_seed, workers=True)
    dataset = data_map[args.dataset](batch_size=args.batch_size, seed=args.data_seed + 42)

    model = model_map[args.model].init_for_dataset(args.dataset, args.model_path)
    explainer = explainer_map[args.explainer](model)

    classifier = FARClassifier(model, explainer, attack_name="ef", candidate_extractor_name=args.candidate_extractor_name,
        rho_max=args.rho_max, rho_batch=args.rho_batch, num_candidates=args.num_c, relax_multilabel_constraint=args.relax_multilabel_constraint,
        stop_word_filter=args.stop_word_filter, random_word_importance=args.random_word_importance, random_synonym=args.random_synonym,
        attribution_loss_name=args.attribution_loss_name, delta=args.expl_delta, optimizer_name=args.optimizer_name, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, datamodule=dataset)

    classifier.print = args.print
    args.monitor_val = "avg_e_loss_v_a"
    args.monitor_mode = "min"
    trainer = get_trainer(args)

    if not args.only_eval:
        trainer.fit(model=classifier, datamodule=dataset, ckpt_path=args.checkpoint_path)
    trainer.test(model=classifier, datamodule=dataset, verbose=True)

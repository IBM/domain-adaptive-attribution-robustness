import os, argparse, json
import numpy as np
from time import time
import torch as ch
import torch.nn.functional as f
import pytorch_lightning as pl
from torchmetrics import functional as tmf

from scripts.base_trainer import BaseClassifier, get_trainer
from models import model_map
from data import data_map
from attacks import attack_map
from constants import WRONG_ORDS
from utils import get_dirname


class AdvClassifier(BaseClassifier):
    def __init__(self, model, datamodule, optimizer_name: str = "adam", learning_rate: float = 0.0001,
            weight_decay: float = 0.0, attack_name: str = "a2t", candidate_extractor_name: str = "distilroberta",
            attack_ratio: float = 0.75, rho_max: float = 0.15, rho_batch: float = 0.0, num_candidates: int = 15,
            stop_word_filter: bool = True, random_word_importance: bool = False, random_synonym: bool = False,
            attack_recall: bool = False):
        super().__init__(model=model, optimizer_name=optimizer_name, learning_rate=learning_rate, weight_decay=weight_decay, datamodule=datamodule)

        self.attack_algo = attack_map[attack_name](candidate_extractor_name=candidate_extractor_name)
        self.dataset_name = datamodule.name
        self.attack_ratio = attack_ratio
        self.rho_max = rho_max
        self.rho_batch = rho_batch
        self.num_candidates = num_candidates
        self.stop_word_filter = stop_word_filter
        self.random_word_importance = random_word_importance
        self.random_synonym = random_synonym
        self.attack_recall = attack_recall

    def training_step(self, batch, batch_idx):
        with ch.no_grad():
            data_sentences, labels = batch
            tokenized = self.tokenizer(data_sentences, max_length=self.model.input_shape[0], device=self.device)
            sentences = self.tokenizer.decode_from_words(tokenized["input_words"], tokenized["word_mask"])
            sentences = ["".join([c for c in s if ord(c) not in WRONG_ORDS]) for s in sentences]

            is_attacked = np.random.random_sample(size=(len(sentences),)) < self.attack_ratio
            attack_start = time()
            if sum(is_attacked) > 0:
                adv_sentences, _, _ = self.attack_algo.attack(sentences=[s for i, s in enumerate(sentences) if is_attacked[i]], model=self.model, labels=labels[is_attacked],
                    multilabel=self.datamodule.multilabel, tokenizer_module=self.tokenizer, embedding_module=self.embedding,
                    device=self.device, rho_max=self.rho_max, rho_batch=self.rho_batch, num_candidates=self.num_candidates,
                    stop_word_filter=self.stop_word_filter, random_word_importance=self.random_word_importance, random_synonym=self.random_synonym,
                    pos_weight=None, attack_recall=self.attack_recall)
            else:
                adv_sentences = []

            attack_time = time() - attack_start
            self.model.zero_grad()

            train_sentences = [adv_sentences.pop(0) if is_attacked[i] else s for i, s in enumerate(sentences)]

            if len(adv_sentences) > 0:
                raise ValueError("Not all adversarial sentences overwritten")
            tokenized = self.tokenizer(train_sentences, max_length=self.model.input_shape[0], device=self.device)
        embeds = self.embedding(tokenized)
        preds = self.model(**embeds)

        if not self.multilabel:
            if self.datamodule.pos_weights is not None:
                loss = f.cross_entropy(preds, labels, weight=self.datamodule.pos_weights.to(preds.device))
            else:
                loss = f.cross_entropy(preds, labels)
        else:
            if self.datamodule.pos_weights is not None:
                loss = f.binary_cross_entropy_with_logits(preds, labels.to(ch.float32), pos_weight=self.datamodule.pos_weights.to(preds.device))
            else:
                loss = f.binary_cross_entropy_with_logits(preds, labels.to(ch.float32))

        # Extract log metrics
        adv_preds, adv_labels = preds[is_attacked], labels[is_attacked]
        clean_preds, clean_labels = preds[~is_attacked], labels[~is_attacked]

        log_dict = {"a_time": float(attack_time / max(sum(is_attacked), 1)), "num_a": float(sum(is_attacked)), "num_c": float(sum(~is_attacked)),
            "loss_tr": loss, "acc_tr_a": tmf.accuracy(adv_preds, adv_labels.to(ch.int32), num_classes=self.model.num_classes) if len(adv_preds) > 0 else 0.0,
            "acc_tr_c": tmf.accuracy(clean_preds, clean_labels.to(ch.int32), num_classes=self.model.num_classes) if len(clean_preds) > 0 else 0.0,
            "f1_tr_a": tmf.f1_score(adv_preds, adv_labels.to(ch.int32), num_classes=self.model.num_classes, average="macro") if len(adv_preds) > 0 else 0.0,
            "f1_tr_c": tmf.f1_score(clean_preds, clean_labels.to(ch.int32), num_classes=self.model.num_classes, average="macro") if len(clean_preds) > 0 else 0.0,}

        if self.multilabel:
            for top_k_r in [0.1, 0.16, 0.3]:
                top_k = int(round(top_k_r * self.model.num_classes))
                if top_k <= self.model.num_classes:
                    log_dict[f"p_{top_k}_tr_a"] = tmf.precision(adv_preds, adv_labels.to(ch.int32), num_classes=self.model.num_classes, top_k=top_k) if len(adv_preds) > 0 else 0.0
                    log_dict[f"p_{top_k}_tr_c"] = tmf.precision(clean_preds, clean_labels.to(ch.int32), num_classes=self.model.num_classes, top_k=top_k) if len(clean_preds) > 0 else 0.0

        # Log metrics
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=False, batch_size=len(data_sentences))
        return loss

    def validation_step(self, batch, batch_idx):
        with ch.no_grad():
            data_sentences, labels = batch
            tokenized = self.tokenizer(data_sentences, max_length=self.model.input_shape[0], device=self.device)
            sentences = self.tokenizer.decode_from_words(tokenized["input_words"], tokenized["word_mask"])
            sentences = ["".join([c for c in s if ord(c) not in WRONG_ORDS]) for s in sentences]
            tokenized = self.tokenizer(sentences, max_length=self.model.input_shape[0], device=self.device)
            embeds = self.embedding(tokenized)
            clean_preds = self.model(**embeds)


        attack_start = time()
        adv_sentences, replacements, max_dist = self.attack_algo.attack(sentences=sentences, labels=labels, model=self.model, multilabel=self.datamodule.multilabel, tokenizer_module=self.tokenizer,
            embedding_module=self.embedding, device=self.device, rho_max=self.rho_max, rho_batch=self.rho_batch, num_candidates=self.num_candidates,
            stop_word_filter=self.stop_word_filter, random_word_importance=self.random_word_importance, random_synonym=self.random_synonym,
            pos_weight=None, attack_recall=self.attack_recall)
        with ch.no_grad():
            adv_tokenized = self.tokenizer(adv_sentences, max_length=self.model.input_shape[0], device=self.device)
            adv_embeds = self.embedding(adv_tokenized)
            adv_preds = self.model(**adv_embeds)

        return clean_preds, adv_preds, labels

    def validation_epoch_end(self, validation_step_outputs, stage_suffix="v"):
        clean_preds = ch.cat([t[0] for t in validation_step_outputs], dim=0)
        adv_preds = ch.cat([t[1] for t in validation_step_outputs], dim=0)
        labels = ch.cat([t[2] for t in validation_step_outputs], dim=0)

        if self.trainer.world_size == 1:
            all_clean_preds = clean_preds
            all_adv_preds = adv_preds
            all_labels = labels
        else:
            all_clean_preds = ch.cat(self.all_gather(clean_preds).unbind(), dim=0)
            all_adv_preds = ch.cat(self.all_gather(adv_preds).unbind(), dim=0)
            all_labels = ch.cat(self.all_gather(labels).unbind(), dim=0)

        if not self.multilabel:
            clean_loss = f.cross_entropy(all_clean_preds, all_labels)
            adv_loss = f.cross_entropy(all_adv_preds, all_labels)
        else:
            clean_loss = f.binary_cross_entropy_with_logits(all_clean_preds, all_labels.to(ch.float32))
            adv_loss = f.binary_cross_entropy_with_logits(all_adv_preds, all_labels.to(ch.float32))

        log_dict = {f"loss_{stage_suffix}_c": clean_loss, f"loss_{stage_suffix}_a": adv_loss }
        for n, metric, _ in self.logged_metrics:
            for avg in ["macro", "micro"]:
                log_dict[f"{n}_{stage_suffix}_{avg}_c"] = self.metrics[f"{n}_{avg}"].to(self.device)(all_clean_preds, all_labels.to(ch.int32))
                log_dict[f"{n}_{stage_suffix}_{avg}_a"] = self.metrics[f"{n}_{avg}"].to(self.device)(all_adv_preds, all_labels.to(ch.int32))

        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True
        )


if __name__ == "__main__":
    with open(os.path.join(get_dirname(__file__), "adv_config.json")) as json_file:
        config = json.load(json_file)
    args = argparse.Namespace(**config)

    pl.seed_everything(args.overall_seed, workers=True)
    dataset = data_map[args.dataset](batch_size=args.batch_size, seed=args.data_seed)

    model = model_map[args.model].init_for_dataset(args.dataset, args.model_path)

    classifier = AdvClassifier(model, datamodule=dataset, attack_name="a2t", candidate_extractor_name=args.candidate_extractor_name,
        rho_max=args.rho_max, rho_batch=args.rho_batch, attack_ratio=args.attack_ratio, num_candidates=args.num_c,
        stop_word_filter=args.stop_word_filter, random_word_importance=args.random_word_importance, random_synonym=args.random_synonym,
        optimizer_name=args.optimizer_name, learning_rate=args.learning_rate, weight_decay=args.weight_decay, attack_recall=args.attack_recall)

    classifier.print = args.print
    if dataset.multilabel and args.attack_recall:
        args.monitor_val = "r_v_macro_a"
        args.monitor_mode = "max"
    else:
        args.monitor_val = "loss_v_a"
        args.monitor_mode = "min"
    trainer = get_trainer(args)

    if not args.only_eval:
        trainer.fit(model=classifier, datamodule=dataset, ckpt_path=args.checkpoint_path)
    trainer.test(model=classifier, datamodule=dataset, verbose=True)

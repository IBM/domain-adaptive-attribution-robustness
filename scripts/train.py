import json, os, argparse
import torch as ch
import torch.nn.functional as f
import pytorch_lightning as pl

from scripts.base_trainer import BaseClassifier, get_trainer
from models import model_map
from data import data_map
from constants import WRONG_ORDS
from utils import get_dirname


class Classifier(BaseClassifier):
    def __init__(self, model, datamodule, optimizer_name="adamw", learning_rate=0.001, weight_decay=0.0):
        super().__init__(model=model, optimizer_name=optimizer_name, learning_rate=learning_rate, weight_decay=weight_decay, datamodule=datamodule)

    def training_step(self, batch, batch_idx):
        data_sentences, labels = batch
        data_sentences = ["".join([c for c in s if ord(c) not in WRONG_ORDS]) for s in data_sentences]

        tokenized = self.tokenizer(data_sentences, max_length=self.model.input_shape[0], device=self.device)
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

        return loss

    def validation_step(self, batch, batch_idx):
        with ch.no_grad():
            data_sentences, labels = batch
            data_sentences = ["".join([c for c in s if ord(c) not in WRONG_ORDS]) for s in data_sentences]
            tokenized = self.tokenizer(data_sentences, max_length=self.model.input_shape[0], device=self.device)
            embeds = self.embedding(tokenized)
            preds = self.model(**embeds)

        return preds, labels

    def validation_epoch_end(self, validation_step_outputs, stage_suffix="v"):
        preds = ch.cat([t[0] for t in validation_step_outputs], dim=0)
        labels = ch.cat([t[1] for t in validation_step_outputs], dim=0)

        if self.trainer.world_size == 1:
            all_preds = preds
            all_labels = labels
        else:
            all_preds = ch.cat(self.all_gather(preds).unbind(), dim=0)
            all_labels = ch.cat(self.all_gather(labels).unbind(), dim=0)

        if not self.multilabel:
            loss = f.cross_entropy(all_preds, all_labels)
        else:
            loss = f.binary_cross_entropy_with_logits(all_preds, all_labels.to(ch.float32))

        log_dict = {f"loss_{stage_suffix}": loss}

        for n, metric, _ in self.logged_metrics:
            for avg in ["macro", "micro"]:
                log_dict[f"{n}_{stage_suffix}_{avg}"] = self.metrics[f"{n}_{avg}"].to(self.device)(all_preds, all_labels.to(ch.int32))

        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)


if __name__ == "__main__":
    with open(os.path.join(get_dirname(__file__), "van_config.json")) as json_file:
        config = json.load(json_file)
    args = argparse.Namespace(**config)

    pl.seed_everything(args.overall_seed, workers=True)
    dataset = data_map[args.dataset](batch_size=args.batch_size, seed=args.data_seed)
    model = model_map[args.model].init_for_dataset(args.dataset, args.model_path)

    classifier = Classifier(model, optimizer_name=args.optimizer_name, learning_rate=args.learning_rate, weight_decay=args.weight_decay, datamodule=dataset)

    classifier.print = args.print
    trainer = get_trainer(args)

    if not args.only_eval:
        trainer.fit(model=classifier, datamodule=dataset, ckpt_path=args.checkpoint_path)
    trainer.test(model=classifier, datamodule=dataset, verbose=True)

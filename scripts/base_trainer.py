import torch as ch
import pytorch_lightning as pl
from collections import OrderedDict
from torchmetrics import Accuracy, F1Score, Precision, Recall

from tokenizer_wrappers import tokenizer_map
from embeddings import embedding_map


class BaseClassifier(pl.LightningModule):
    def __init__(self, model, optimizer_name: str = "adamw", learning_rate: float = 0.001, weight_decay: float = 0.0, datamodule=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer_map[self.model.tokenizer_name]()
        cls_embeds = None
        if self.model.word_embeddings_var_name is not None:
            cls_embeds = getattr(self.model, self.model.word_embeddings_var_name[0])
            for attr in self.model.word_embeddings_var_name[1:]:
                cls_embeds = getattr(cls_embeds, attr)
        self.embedding = embedding_map[model.embedding_name](word_embeddings=cls_embeds)
        self.datamodule = datamodule

        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.multilabel = datamodule.multilabel

        self.logged_metrics = [("acc", Accuracy, {}), ("f1", F1Score, {}), ("r", Recall, {}), (f"p1", Precision, {"top_k": 1})]
        if self.multilabel:
            self.logged_metrics += [(f"p{int(round(0.16 * self.model.num_classes))}", Precision, {"top_k": int(round(0.16 * self.model.num_classes))}),
                    (f"p{int(round(0.3 * self.model.num_classes))}", Precision, {"top_k": int(round(0.3 * self.model.num_classes))})]

        self.metrics = {}
        for n, metric, additional_init_args in self.logged_metrics:
            for avg in ["macro", "micro"]:
                if self.datamodule.multilabel:
                    setattr(self, f"{n}_{avg}", metric(task="multilabel", num_labels=self.model.num_classes, average=avg, num_classes=self.model.num_classes, **additional_init_args))
                    self.metrics[f"{n}_{avg}"] = metric(task="multilabel", num_labels=self.model.num_classes, average=avg, num_classes=self.model.num_classes, **additional_init_args)
                else:
                    setattr(self, f"{n}_{avg}", metric(num_classes=self.model.num_classes, average=avg, **additional_init_args))
                    self.metrics[f"{n}_{avg}"] = metric(num_classes=self.model.num_classes, average=avg, **additional_init_args)

    def configure_optimizers(self):
        trained_params = self.model.parameters()
        if self.optimizer_name == "adam":
            optimizer = ch.optim.Adam(trained_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adamw":
            optimizer = ch.optim.AdamW(trained_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            optimizer = ch.optim.SGD(trained_params, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9, dampening=0.99)
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_name} not implemented.")

        return optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = OrderedDict([(k.removeprefix("model."), v) for k, v in checkpoint["state_dict"].items()])

    def on_load_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = OrderedDict([(f"model.{k}", v) if k in self.model.state_dict() else (k, v) for k, v in checkpoint["state_dict"].items()])

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs, stage_suffix="te")


def get_trainer(args):
    callbacks = []
    if args.swa_lr is not None:
        callbacks.append(pl.callbacks.StochasticWeightAveraging(swa_lrs=args.swa_lr))
    callbacks.append(
        pl.callbacks.ModelCheckpoint(monitor=args.monitor_val, save_top_k=3, save_last=True, verbose=True, save_weights_only=False, mode=args.monitor_mode, every_n_epochs=1, auto_insert_metric_name=True)
    )
    callbacks.append(
        pl.callbacks.RichProgressBar(
            theme=pl.callbacks.progress.rich_progress.RichProgressBarTheme(description="green_yellow",progress_bar="green1",
            progress_bar_finished="green1", progress_bar_pulse="#6206E0", batch_progress="green_yellow", time="grey82",
            processing_speed="grey82", metrics="grey82")))
    return pl.Trainer(accelerator='gpu', enable_progress_bar=True, accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.epochs, limit_test_batches=args.num_test_batches, log_every_n_steps=args.log_every_n_steps,
        deterministic=True, gradient_clip_val=args.gradient_clip_val, profiler=args.profiler, default_root_dir=args.exp_folder,
        callbacks=callbacks, fast_dev_run=args.fast_dev_run, precision=args.precision, amp_backend=args.amp_backend)

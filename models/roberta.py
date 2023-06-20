import torch as ch
from transformers import RobertaModel
from . import BaseModel
from data import data_length_map, data_num_classes


class RoBERTa(BaseModel):
    attention_layer_idx = -1
    attention_head_idx = -1
    attention_out_idx = 0

    tokenizer_name = "roberta"
    embedding_name = "roberta"

    def __init__(self, input_shape, num_classes, lin_dims, attention_layer_idx, attention_head_idx, attention_out_idx,
                 weight_path=None):
        super().__init__(input_shape=input_shape, num_classes=num_classes)

        self.lin_dims = lin_dims
        self.attention_layer_idx = attention_layer_idx
        self.attention_head_idx = attention_head_idx
        self.attention_out_idx = attention_out_idx

        self.cls_model = RobertaModel.from_pretrained("roberta-base", output_attentions=True)
        self.word_embeddings_var_name = ["cls_model", "embeddings", "word_embeddings"]

        tmp_lin_dims = [768] + self.lin_dims + [self.num_classes]
        self.fcs = ch.nn.ModuleList([ch.nn.Linear(tmp_lin_dims[i - 1], tmp_lin_dims[i]) for i in range(
            1, len(tmp_lin_dims))])

        if weight_path is not None:
            self._load_weights(weight_path)

    def _forward_preprocessing(self, word_embeds, token_type_ids):
        embeds = self.cls_model.embeddings(input_ids=None, position_ids=None, token_type_ids=token_type_ids,
                                           inputs_embeds=word_embeds, past_key_values_length=0)
        return embeds

    def _forward_features(self, input_embeds, attention_mask, token_type_ids):
        outs = self.cls_model(inputs_embeds=input_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outs

    def _forward_classifier(self, x):
        for lin_layer in self.fcs[:-1]:
            x = lin_layer(x)
        return self.fcs[-1](x)

    def forward(self, input_embeds, attention_mask, token_type_ids, return_att_weights=False):
        full_embeds = self._forward_preprocessing(input_embeds, token_type_ids)
        outs = self._forward_features(full_embeds, attention_mask, token_type_ids)

        if return_att_weights:
            return self._forward_classifier(outs.pooler_output), outs.attentions[self.attention_layer_idx][
                :, self.attention_head_idx, self.attention_out_idx]
        return self._forward_classifier(outs.pooler_output)

    @staticmethod
    def init_for_dataset(dataset: str, weight_path: str = None):
        return RoBERTa(input_shape=(min(512, data_length_map.get(dataset, 512)),), num_classes=data_num_classes[dataset],
                       lin_dims=[], attention_layer_idx=-1, attention_head_idx=-1, attention_out_idx=0, weight_path=weight_path)
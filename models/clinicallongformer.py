import torch as ch
from transformers import LongformerModel
from . import BaseModel
from data import data_length_map, data_num_classes


# Training should be inspired by: https://aclanthology.org/2021.emnlp-main.481.pdf
class ClinicalLongformer(BaseModel):
    attention_layer_idx = None
    attention_head_idx = None
    attention_out_idx = None

    tokenizer_name = "clinicallongformer"
    embedding_name = "clinicallongformer"

    def __init__(self, input_shape, num_classes, lin_dims, attention_layer_idx, attention_head_idx, attention_out_idx, weight_path=None):
        super().__init__(input_shape=input_shape, num_classes=num_classes)

        self.lin_dims = lin_dims
        self.attention_layer_idx = attention_layer_idx
        self.attention_head_idx = attention_head_idx
        self.attention_out_idx = attention_out_idx

        self.cls_model = LongformerModel.from_pretrained("yikuan8/Clinical-Longformer", output_attentions=True)
        self.word_embeddings_var_name = ["cls_model", "embeddings", "word_embeddings"]

        tmp_lin_dims = [768] + self.lin_dims + [self.num_classes]
        self.fcs = ch.nn.ModuleList([ch.nn.Linear(tmp_lin_dims[i - 1], tmp_lin_dims[i]) for i in range(1, len(tmp_lin_dims))])

        if weight_path is not None:
            self._load_weights(weight_path)

    def _forward_preprocessing(self, word_embeds, token_type_ids):
        embeds = self.cls_model.embeddings(input_ids=None, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=word_embeds)
        return embeds

    def _forward_features(self, input_embeds, attention_mask, token_type_ids, global_attention_mask=None):
        tmp_det = ch.backends.cudnn.deterministic
        ch.use_deterministic_algorithms(False, warn_only=True)
        outs = self.cls_model(inputs_embeds=input_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids, global_attention_mask=global_attention_mask)

        ch.use_deterministic_algorithms(tmp_det, warn_only=True)
        return outs

    def _forward_classifier(self, x):
        for lin_layer in self.fcs[:-1]:
            x = lin_layer(x)
        return self.fcs[-1](x)

    def forward(self, input_embeds, attention_mask, token_type_ids, global_attention_mask=None, return_att_weights=False):
        full_embeds = self._forward_preprocessing(input_embeds, token_type_ids)
        outs = self._forward_features(input_embeds=full_embeds,attention_mask=attention_mask, token_type_ids=token_type_ids, global_attention_mask=global_attention_mask)

        if return_att_weights:
            return self._forward_classifier(outs.pooler_output), outs.global_attentions[self.attention_layer_idx][:, self.attention_head_idx, :, 0]
        return self._forward_classifier(outs.pooler_output)

    @staticmethod
    def init_for_dataset(dataset: str, weight_path: str = None):
        return ClinicalLongformer(input_shape=(min(4096, data_length_map.get(dataset, 4096)),), num_classes=data_num_classes[dataset], lin_dims=[], attention_layer_idx=-1, attention_head_idx=-1, attention_out_idx=0, weight_path=weight_path)

import torch as ch
from transformers import AutoModel


class BioLinkBERTEmbedding(ch.nn.Module):
    def __init__(self, word_embeddings=None):
        super().__init__()
        if word_embeddings is None:
            self.word_embeddings = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-base").embeddings.word_embeddings
        else:
            self.word_embeddings = word_embeddings

    def __call__(self, tokenized):
        embedding_output = self.word_embeddings(tokenized["tokenized"]["input_ids"])

        return {"input_embeds": embedding_output, "token_type_ids": tokenized["tokenized"]["token_type_ids"], "attention_mask": tokenized["tokenized"]["attention_mask"]}

# Download model from https://github.com/uf-hobi-informatics-lab/2019_N2C2_Track1_ClinicalSTS/tree/c00c5b822105a9a4c36ae7fcc268f8f25b01741c
import os
import transformers
from utils import get_project_rootdir
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch as ch


class MedBCE(ch.nn.Module):
    def __init__(self, **kwargs):
        super(MedBCE, self).__init__()

        self.max_length = 4096  # 4096 is the longest input to any model
        self.chunk_size = 128  # MedSTS has a mean word count of ~48, therefore 128 seems a good chunk size

        self.name_tag = os.path.join(get_project_rootdir(), "sentence_encoders", "models",
                                     "2019n2c2_tack1_roberta_stsc_6b_16b_3c_8c")

        self.tokenizer = RobertaTokenizer.from_pretrained(self.name_tag)
        self.model = RobertaForSequenceClassification.from_pretrained(self.name_tag)

    def semantic_sim(self, sentence1, sentence2, reduction="min", **kwargs):
        transformers.logging.set_verbosity_error()
        sent1_tok = self.tokenizer([sentence1], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        sent2_tok = self.tokenizer([sentence2], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        sentences1 = [self.tokenizer.decode(sent1_tok["input_ids"][0][self.chunk_size*_i: self.chunk_size*(_i+1)], skip_special_tokens=True, clean_up_tokenization_spaces=True) for _i in range(int((self.max_length + self.chunk_size - 1)/self.chunk_size))]
        sentences1 = [s for s in sentences1 if s != ""]
        sentences2 = [self.tokenizer.decode(sent2_tok["input_ids"][0][self.chunk_size*_i: self.chunk_size*(_i+1)], skip_special_tokens=True, clean_up_tokenization_spaces=True) for _i in range(int((self.max_length + self.chunk_size - 1)/self.chunk_size))]
        sentences2 = [s for s in sentences2 if s != ""]

        sims = []
        for i in range(min(len(sentences1), len(sentences2))):
            tokenized = self.tokenizer([(sentences1[i], sentences2[i])], max_length=self.chunk_size, padding="max_length", truncation=True, return_tensors="pt")
            with ch.no_grad():
                sims.append(self.model(**(tokenized.to(list(self.model.parameters())[0].device))).logits.clamp(min=0.0, max=5.0).div(5.0))

        transformers.logging.set_verbosity_warning()
        if reduction == "mean" or reduction is None:
            return ch.mean(ch.cat(sims))
        elif reduction == "min":
            return ch.min(ch.cat(sims))
        else:
            raise ValueError(f"Reduction '{reduction}' not supported")

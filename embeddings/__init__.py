from .roberta import RoBERTaEmbedding
from .biolinkbert import BioLinkBERTEmbedding
from .clinicallongformer import ClinicalLongformerEmbedding

embedding_map = {"roberta": RoBERTaEmbedding, "biolinkbert": BioLinkBERTEmbedding, "clinicallongformer": ClinicalLongformerEmbedding}
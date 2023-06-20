import os
from .roberta import RoBERTaTokenizer
from .distilroberta import DistilRoBERTaTokenizer
from .pubmedbert import PubMedBERTTokenizer
from .biolinkbert import BioLinkBERTTokenizer
from .clinicallongformer import ClinicalLongformerTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "true"


tokenizer_map = {"roberta": RoBERTaTokenizer, "distilroberta": DistilRoBERTaTokenizer, "biolinkbert": BioLinkBERTTokenizer,
    "pubmedbert": PubMedBERTTokenizer, "clinicallongformer": ClinicalLongformerTokenizer}

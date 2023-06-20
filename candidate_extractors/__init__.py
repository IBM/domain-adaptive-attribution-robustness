from .distilroberta_mlm import DistilRoBERTaMLM
from .pubmedbert_mlm import PubMedBERTMLM
from .clinicallongformer_mlm import ClinicalLongformerMLM


candidate_extractor_map = {"distilroberta": DistilRoBERTaMLM, "pubmedbert": PubMedBERTMLM, "clinicallongformer": ClinicalLongformerMLM}

from .datamodules import HOCDataModule
from .datamodules import DrugReviewsDataModule
from .datamodules import MIMICDataModule

data_map = {HOCDataModule.name: HOCDataModule, DrugReviewsDataModule.name: DrugReviewsDataModule, MIMICDataModule.name: MIMICDataModule}

data_length_map = {HOCDataModule.name: 256, DrugReviewsDataModule.name: 128, MIMICDataModule.name: 4096}

data_num_classes = {HOCDataModule.name: 10, DrugReviewsDataModule.name: 5, MIMICDataModule.name: 50}

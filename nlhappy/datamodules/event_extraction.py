from .base import PLMDataModule



class EventExtractionDataModule(PLMDataModule):
    def __init__(self, 
                dataset: str, 
                plm: str, 
                max_length: int, 
                batch_size: int, 
                pin_memory: bool, 
                num_workers: int, 
                data_dir: str = './datasets', 
                pretrained_dir: str = './plms'):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        pass

    def transform(self, batch):
        pass 
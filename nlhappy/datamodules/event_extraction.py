import pytorch_lightning as pl



class EventExtractionDataModule(pl.LightningDataModule):
    def __init__(self, 
                dataset: str, 
                plm: str, 
                max_length: int, 
                batch_size: int, 
                pin_memory: bool, 
                num_workers: int, 
                dataset_dir: str = './datasets', 
                plm_dir: str = './plms'):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        pass

    def transform(self, batch):
        pass 
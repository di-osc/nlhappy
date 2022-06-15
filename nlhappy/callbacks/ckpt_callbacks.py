from pytorch_lightning import Callback
import torch
import os
import logging

log = logging.getLogger(__name__)


class LoadModelStateDict(Callback):
    def __init__(self, ckpt_path: str) -> None:
        super().__init__()
        self.ckpt_path = ckpt_path
        
    def on_train_start(self, trainer, pl_module) -> None:
        state_dict = torch.load(self.ckpt_path)['state_dict']
        log.info((f'Loading state from {self.ckpt_path} on device {pl_module.device}'))
        pl_module.load_state_dict(state_dict)
        log.info('all weights loaded')
        

class LoadPLMStateDict(Callback):
    """Load model from checkpoint."""

    def on_train_start(self, trainer, pl_module):
        ckpt_path = os.path.join(pl_module.hparams.plm_dir, pl_module.hparams.plm, 'pytorch_model.bin')
        log.info((f'Loading PLM state dict from {ckpt_path} on device {pl_module.device}'))
        try: 
            pl_module.plm.load_state_dict(torch.load(ckpt_path))
        except Exception:
            pass
        try: 
            pl_module.bert.load_state_dict(torch.load(ckpt_path))
        except Exception:
            pass
        try: 
            pl_module.encoder.load_state_dict(torch.load(ckpt_path))
        except Exception:
            pass
        try:
            pl_module.decoder.load_state_dict(torch.load(ckpt_path))
        except Exception:
            pass
        log.info('all weights loaded')
        
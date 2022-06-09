from pytorch_lightning import Callback
import torch
import os
import logging

log = logging.getLogger(__name__)



class LoadPLM(Callback):
    """Load model from checkpoint."""

    def on_train_start(self, trainer, pl_module):
        ckpt_path = os.path.join(pl_module.hparams.plm_dir, pl_module.hparams.plm, 'pytorch_model.bin')
        log.info((f'Loading PLM from {ckpt_path} on device {pl_module.device}'))
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
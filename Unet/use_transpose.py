
from .train.checkpoint import load_checkpoint
from .model.unet_transpose import Unet
import torch

def load_unet_transpose(path: str, device=torch.device("cpu")):
    unet = Unet()
    checkpoint = load_checkpoint(path, device)
    unet.load_state_dict(checkpoint["model"])
    unet.to(device)
    unet.eval()
    return unet



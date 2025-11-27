from .egodex import EgoDexDataset
from .oxe import OXEDataset
from .agibotworld import AgiBotWorldDataset
from .holoassist import HoloAssistDataset

DATASETS = {
    "egodex": EgoDexDataset,
    "oxe": OXEDataset,
    "agibotworld": AgiBotWorldDataset,
    "holoassist": HoloAssistDataset,
}

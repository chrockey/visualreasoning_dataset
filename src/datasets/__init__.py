from .egodex import EgoDexDataset
from .oxe import OXEDataset

DATASETS = {
    "egodex": EgoDexDataset,
    "oxe": OXEDataset,
    # "agibotworld": AgiBotWorldDataset,
    # "holoassist": HoloAssistDataset,
}

from denoisplit.core.custom_enum import Enum


class SamplerType(Enum):
    DefaultSampler = 0
    RandomSampler = 1
    SingleImgSampler = 2
    NeighborSampler = 3
    ContrastiveSampler = 4
    DefaultGridSampler = 5
    IntensityAugSampler = 6
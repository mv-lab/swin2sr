from denoisplit.core.custom_enum import Enum


class MixedInputType(Enum):
    # aligned means that mixed input has the same distribution as in reality: it is not the case that any two
    # random images from the two crops are mixed to create this mixed input. Instead only co-located channels are mixed.
    Aligned = 'aligned'
    Randomized = 'randomized'
    # this means that the mixed input is simply the average of the individual channels
    ConsistentWithSingleInputs = 'consistent_with_single_inputs'

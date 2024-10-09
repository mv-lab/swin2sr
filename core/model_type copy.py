from denoisplit.core.custom_enum import Enum


class ModelType(Enum):
    LadderVae = 3
    LadderVaeTwinDecoder = 4
    LadderVAECritic = 5
    # Separate vampprior: two optimizers
    LadderVaeSepVampprior = 6
    # one encoder for mixed input, two for separate inputs.
    LadderVaeSepEncoder = 7
    LadderVAEMultiTarget = 8
    LadderVaeSepEncoderSingleOptim = 9
    UNet = 10
    BraveNet = 11
    LadderVaeStitch = 12
    LadderVaeSemiSupervised = 13
    LadderVaeStitch2Stage = 14  # Note that previously trained models will have issue.
    # since earlier, LadderVaeStitch2Stage = 13, LadderVaeSemiSupervised = 14
    LadderVaeMixedRecons = 15
    LadderVaeCL = 16
    LadderVaeTwoDataSet = 17  #on one subdset, apply disentanglement, on other apply reconstruction
    LadderVaeTwoDatasetMultiBranch = 18
    LadderVaeTwoDatasetMultiOptim = 19
    LVaeDeepEncoderIntensityAug = 20
    AutoRegresiveLadderVAE = 21
    LadderVAEInterleavedOptimization = 22
    Denoiser = 23
    DenoiserSplitter = 24
    SplitterDenoiser = 25
    LadderVAERestrictedReconstruction = 26
    LadderVAETwoDataSetRestRecon = 27
    LadderVAETwoDataSetFinetuning = 28

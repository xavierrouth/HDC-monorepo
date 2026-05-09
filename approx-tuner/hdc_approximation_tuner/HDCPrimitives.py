from abc import ABC, abstractmethod
from enum import IntEnum
import json

class HDCOperation(IntEnum):
    HDCMatmul = 1
    HDCHammingDistance = 2
    HDCCosineSimilarity = 3
    HDCL2Norm = 4
    HDCTrain = 5
    HDCTrainLoop = 6
    HDCInference = 7
    HDCInferenceLoop = 8





hdc_op_str_to_enum = {
    'llvm.hpvm.hdc.matmul' : HDCOperation.HDCMatmul,
    'llvm.hpvm.hdc.hamming.distance' : HDCOperation.HDCHammingDistance,
    'llvm.hpvm.hdc.cossim' : HDCOperation.HDCCosineSimilarity,
    "llvm.hpvm.hdc.l2norm" : HDCOperation.HDCL2Norm,
    "llvm.hpvm.hdc.training" : HDCOperation.HDCTrain,
    "llvm.hpvm.hdc.training.loop" : HDCOperation.HDCTrainLoop,
    "llvm.hpvm.hdc.inference" : HDCOperation.HDCInference,
    "llvm.hpvm.hdc.inference.loop" : HDCOperation.HDCInferenceLoop,
}

# Inverse map
hdc_enum_to_op_str = {value : key for (key,value) in hdc_op_str_to_enum.items()}

def get_reducing_dim(Op, return_type, argument_types):

    if Op in [
            HDCOperation.HDCCosineSimilarity,
            HDCOperation.HDCHammingDistance,
            HDCOperation.HDCL2Norm,
            HDCOperation.HDCMatmul,
    ]:
        return argument_types[0].get_encoding_dimension()
    else:
        assert False, "Unsupported Op"

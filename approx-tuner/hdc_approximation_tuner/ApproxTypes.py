from .Types import ElementType, HyperVector, HyperMatrix, isIntegerType
from .HDCPrimitives import HDCOperation
from .PCMDesc import PCMDesc
from abc import ABC, abstractmethod
from typing import List
import opentuner
from opentuner.search.manipulator import (ConfigurationManipulator,
                                          IntegerParameter,
                                          LogIntegerParameter,
                                          SelectorParameter,
                                          SwitchParameter,
                                          EnumParameter,
                                          PowerOfTwoParameter,
                                          BooleanParameter,
                                          FloatParameter
                                          )



from pyatf import TP, Interval, Set

MLC_PARAM_NAME = "MLC"
ADC_RESOLUTION_NAME = "ADC_RES"
ADC_STEP_NAME = "ADC_STEP"
ADC_FLOOR_NAME = "ADC_FLOOR"
ADC_CEIL_NAME = "ADC_CEIL"
WRITE_VERIFY_CYCLE_NAME = "WRITE_VERIFY_CYCLE"

class ApproxType(ABC):
    def __init__(self, name : str, desc : str, is_global : bool):
        self.name = name
        self.desc = desc
        self.is_global = is_global

    def __str__(self):
        return f"ApproxType Object - Name: {self.name}, Desc: {self.desc}"


    @abstractmethod
    def isSoftwareApproximation(self) -> bool:
        pass

    def isHardwareApproximation(self) -> bool:
        return not self.isSoftwareApproximation()

    @abstractmethod
    def get_pyatf_parameters(self):
        pass

    @abstractmethod
    def get_opentuner_parameters(self):
        pass


class ApproxEncodingDimension(ApproxType):
    def __init__(self,  minimum_dim : int, maximum_dim : int, only_powers_of_2 : bool):
        super().__init__("GLOBAL_EncodingDim", "Change the encoding dimension globally for the HDC Application", True)

        # Minimum and Maximum dims are inclusive
        self.minimum_dim = minimum_dim
        self.maximum_dim = maximum_dim
        self.only_powers_of_2 = only_powers_of_2

    @classmethod
    def isApplicableOnHDCOp(cls, op : HDCOperation) -> bool:
        # Approximating the encoding dimension affects applications globally
        return False

    def isSoftwareApproximation(self) -> bool:
        return True

    def get_range_of_dimensions(self):
        if not self.only_powers_of_2:
            return range(self.minimum_dim, self.maximum_dim + 1)

        # If using only powers of 2

        powers = []
        n = 1

        while n < self.minimum_dim:
            n *= 2

        while n <= self.maximum_dim:
            powers.append(n)
            n *= 2

        return powers

    def __str__(self):
        return f"ApproxEncodingDimension Object - Minimum : {self.minimum_dim}, Maximum : {self.maximum_dim}, Only Powers of 2: {self.only_powers_of_2}"


    def get_pyatf_parameters(self):
        if self.only_powers_of_2:
            return [TP(self.name, Interval(self.minimum_dim, self.maximum_dim) , lambda Enc:  Enc > 0 and (Enc & (Enc - 1)) == 0 )]
        else:
            return [TP(self.name, Interval(self.minimum_dim, self.maximum_dim))]

    def get_opentuner_parameters(self):
        if self.only_powers_of_2:
            return [PowerOfTwoParameter(self.name,self.minimum_dim, self.maximum_dim)]
        else:
            return [IntegerParameter(self.name, self.minimum_dim, self.maximum_dim)]


class ApproxElementType(ApproxType):
    def __init__(self, name : str,  srcElemTy : ElementType):
        super().__init__(name, "Approximate the representation of the element type of hypervector/matrices", True)
        self.srcElemTy = srcElemTy


    @classmethod
    def isApplicableOnHDCOp(cls, op : HDCOperation) -> bool:
        return True

    def isSoftwareApproximation(self) -> bool:
        return True

    def get_range_element_type(self):
        types = [ty for ty in ElementType if ty.value <= self.srcElemTy.value]
        return types

    def get_pyatf_parameters(self):
        return [TP(self.name, Set(*self.get_range_element_type()))]

    def get_opentuner_parameters(self):
        return [EnumParameter(self.name, self.get_range_element_type())]



class ApproxAutoBinarize(ApproxType):
    def __init__(self):
        name = "GLOBAL_AutoBinarize"
        super().__init__(name, "Automatic Binarization Transformation for HDC operations", True)


    @classmethod
    def isApplicableOnHDCOp(cls, op : HDCOperation) -> bool:
        return False

    def isSoftwareApproximation(self) -> bool:
        return True

    def get_pyatf_parameters(self):
        return [TP(self.name, Set('True', 'False'))]

    def get_opentuner_parameters(self):
        return [BooleanParameter(self.name)]

class ApproxAlgorithmicChoiceBase(ApproxType):
    def __init__(self,name,  variants : List[HDCOperation]):
        super().__init__(name, "Replace HDC Operation with more efficient / more approximate HDC operations", False)
        self.variants = variants


    @classmethod
    def isApplicableOnHDCOp(self, op : HDCOperation) -> bool:
        return False


    def isSoftwareApproximation(self) -> bool:
        return True

    def get_pyatf_parameters(self):
        return [TP(self.name+"+ALGChoice", Set(*self.variants))]

    def get_opentuner_parameters(self):
        return [EnumParameter(self.name+"+ALGChoice", self.variants)]


class ApproxAlgorithmicChoiceHammingDistance(ApproxAlgorithmicChoiceBase):
    def __init__(self,name):
        super().__init__(name,
                         [HDCOperation.HDCHammingDistance,
                          HDCOperation.HDCCosineSimilarity,
                          ]
                         )

    @classmethod
    def isApplicableOnHDCOp(self, op : HDCOperation) -> bool:
        return op in [
            HDCOperation.HDCHammingDistance,
            HDCOperation.HDCCosineSimilarity,
                          ]



class ApproxReductionPerforation(ApproxType):
    def __init__(self, name , reducing_dim_extent : int, use_float = True, tune_end_only = False):
        super().__init__(name, "Compute result of HDC operation perforated according to the perforation parameters of base, stride and extent", False)
        self.reducing_dim_extent = reducing_dim_extent
        self.base_name = self.name +"+" + "RedPerfBase"
        self.stride_name = self.name +"+" + "RedPerfStride"
        self.extent_name = self.name +"+" + "RedPerfExtent"
        self.use_float = use_float
        self.tune_end_only = tune_end_only


    @classmethod
    def isApplicableOnHDCOp(cls, op : HDCOperation) -> bool:
        return op in [
            HDCOperation.HDCCosineSimilarity,
            HDCOperation.HDCHammingDistance,
            HDCOperation.HDCMatmul,
        ]

    def isSoftwareApproximation(self) -> bool:
        return True

    def get_non_approximated_perforation_params(self):
        return (0, 1, self.reducing_dim_extent)


    def get_pyatf_parameters(self):
        if self.tune_end_only:
            return [
                TP(self.base_name, Interval(0, 0)),
                TP(self.stride_name, Interval(1, 1)),
                TP(self.extent_name, Interval(1, self.reducing_dim_extent)),
            ]
        return [
            TP(self.base_name, Interval(0, self.reducing_dim_extent)),
            TP(self.stride_name, Interval(1, self.reducing_dim_extent), lambda stride: stride > 0 and (stride & (stride - 1)) == 0 ),
            TP(self.extent_name, Interval(1, self.reducing_dim_extent)),
        ]

    def get_opentuner_parameters(self):
        if self.tune_end_only:
            return [
                FloatParameter(self.base_name, 0.0, 0.0),
                IntegerParameter(self.stride_name, 1, 1),
                FloatParameter(self.extent_name, 0.0, 1.0),
            ]
        if self.use_float:
            return [
                FloatParameter(self.base_name, 0.0, 0.0),
                IntegerParameter(self.stride_name, 1, self.reducing_dim_extent),
                FloatParameter(self.extent_name, 0.0, 1.0),
            ]
        else:
            return [
                IntegerParameter(self.base_name, 0, self.reducing_dim_extent),
                IntegerParameter(self.stride_name, 1, self.reducing_dim_extent),
                IntegerParameter(self.extent_name, 0, self.reducing_dim_extent),
            ]



# Hardware Approximation for Phase Change Memory HDC accelerator
class ApproxPCM(ApproxType):
    def __init__(self, hw_desc : PCMDesc ):
        super().__init__("GLOBAL_PCM", "Hardware approximation for PCM", True)
        self.hw_desc = PCMDesc


    def isSoftwareApproximation(self) -> bool:
        return False

    @classmethod
    def isApplicableOnHDCOp(cls, op : HDCOperation) -> bool:
        return op in [
            HDCOperation.HDCTrainLoop,
            HDCOperation.HDCInferenceLoop,
        ]




    def get_pyatf_parameters(self):

        param_descs = self.hw_desc.get_tuning_param_desc()

        parameters = []
        for (param_name, min_val, max_val) in param_descs:
            param = TP(self.name+"_"+param_name,
                       Interval(min(min_val, max_val), max(min_val, max_val))
                       )
            parameters.append(param)


        return parameters

    def get_opentuner_parameters(self):
        # breakpoint()
        # param_descs = self.hw_desc.get_tuning_param_desc()

        # FIXME: Change this for pruning.
        param_descs = [
            (MLC_PARAM_NAME, 1, 4 ),
            (ADC_RESOLUTION_NAME, 1, 10),
            (ADC_STEP_NAME, 1, 10),
            (ADC_FLOOR_NAME, -15, 0),
            (ADC_CEIL_NAME, 0, 16),
            (WRITE_VERIFY_CYCLE_NAME, 0, 7),
        ]

        parameters = []
        for (param_name, min_val, max_val) in param_descs:
            param = IntegerParameter(self.name+"_"+param_name,
                                     min(min_val, max_val),
                                     max(min_val, max_val)
                                     )
            parameters.append(param)


        return parameters




# Flexible tuning parameter for application specific params
# for paramaeters which map to enumerable types
class ApproxCustomEnum(ApproxType):
    def __init__(self, name, num_cases = 1):
        param_name = "CUSTOM+" + name
        self.num_cases = num_cases
        super().__init__(param_name, "Application specific Enum parameter", True)


    @classmethod
    def isApplicableOnHDCOp(cls, op : HDCOperation) -> bool:
        return False

    def isSoftwareApproximation(self) -> bool:
        return True

    def get_pyatf_parameters(self):
        cases = [i for i in range(self.num_cases)]
        return [TP(self.name, Set(*cases))]

    def get_opentuner_parameters(self):
        cases = [i for i in range(self.num_cases)]
        return [EnumParameter(self.name, cases)]


class ApproxCustomInteger(ApproxType):
    def __init__(self, name, num_cases = 1):
        param_name = "CUSTOM+" + name
        self.num_cases = num_cases
        super().__init__(param_name, "Application specific integer parameter", True)


    @classmethod
    def isApplicableOnHDCOp(cls, op : HDCOperation) -> bool:
        return False

    def isSoftwareApproximation(self) -> bool:
        return True

    def get_pyatf_parameters(self):
        return [TP(self.name, Interval(0, self.num_cases))]

    def get_opentuner_parameters(self):
        return [IntegerParameter(self.name, 0, self.num_cases)]


class ApproxCustomFloat(ApproxType):
    def __init__(self, name, min_val: float = 0.0, max_val: float = 1.0):
        param_name = "CUSTOM+" + name
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(param_name, "Application specific float parameter", True)

    @classmethod
    def isApplicableOnHDCOp(cls, op: HDCOperation) -> bool:
        return False

    def isSoftwareApproximation(self) -> bool:
        return True

    def get_pyatf_parameters(self):
        return [TP(self.name, Interval(self.min_val, self.max_val))]

    def get_opentuner_parameters(self):
        return [FloatParameter(self.name, self.min_val, self.max_val)]

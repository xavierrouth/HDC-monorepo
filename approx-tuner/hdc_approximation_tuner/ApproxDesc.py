from .Types import ElementType, HyperVector, HyperMatrix
from .HDCPrimitives import HDCOperation
from .ApproxTypes import *
from typing import List


class HDCOperationHandle:
    def __init__(self, CallingFunction : str, ID : int, HDCOpName : str, op_type: str):
        self.CallingFunction = CallingFunction
        self.ID = ID
        self.HDCOpName = HDCOpName
        self.op_type = op_type
    def get_handle_name(self):
        return "+".join([self.CallingFunction, self.HDCOpName, str(self.ID), self.op_type])


class ApproxDesc:
    def __init__(self, is_global : bool , approximation_tys : List[ApproxType], CallingContext : HDCOperationHandle):
        self.is_global = is_global
        self.approximation_types = approximation_tys
        self.CallingContext = CallingContext

    def get_opentuner_parameters(self):
        params = []
        for apx_ty in self.approximation_types:
            params += apx_ty.get_opentuner_parameters()
        return params

    def get_pyatf_parameters(self):
        params = []
        for apx_ty in self.approximation_types:
            params += apx_ty.get_pyatf_parameters()
        return params











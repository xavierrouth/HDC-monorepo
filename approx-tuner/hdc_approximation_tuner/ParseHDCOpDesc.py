from .ApproxDesc import HDCOperationHandle, ApproxDesc
from .Types import HDCOperationKind, ElementType, type_dict, HyperVector, HyperMatrix
from .HDCPrimitives import HDCOperation, hdc_op_str_to_enum, get_reducing_dim

from .ApproxTypes import *






def parse_element_type(type_str):
    if '<' in type_str:
        type_str = type_str.split("x")[-1].split(">")[0].strip()

    assert type_str in type_dict
    return type_dict[type_str]



def parse_vector_type(type_str):
    assert "<" in type_str
    num_elements = type_str.split("x")[0].split("<")[1].strip()
    num_elements = int(num_elements)

    element_type_str =  type_str.split("x")[1].split(">")[0].strip()
    assert element_type_str in type_dict
    element_type = type_dict[element_type_str]
    return HyperVector(element_type, num_elements)


def parse_matrix_type(type_str):
    assert "<" in type_str
    num_elements_rows = type_str.split("x")[0].split("<")[1].strip()
    num_elements_rows = int(num_elements_rows)

    num_elements_cols = type_str.split("x")[1].strip()
    num_elements_cols = int(num_elements_cols)

    element_type_str =  type_str.split("x")[-1].split(">")[0].strip()
    assert element_type_str in type_dict
    element_type = type_dict[element_type_str]
    return HyperMatrix(element_type, num_elements_rows, num_elements_cols)

def is_scalar_type(type_str: str):
    return "<" not in type_str

def is_vector_type(type_str : str):
    return "<"  in type_str and type_str.count("x") == 1

def is_matrix_type(type_str : str):
    return "<"  in type_str and type_str.count("x") == 2


def parse_type(arg_type_str : str):
    arg_type = None
    if is_scalar_type(arg_type_str):
        arg_type = parse_element_type(arg_type_str)
    elif is_vector_type(arg_type_str):
        arg_type = parse_vector_type(arg_type_str)
    elif is_matrix_type(arg_type_str):
        arg_type = parse_matrix_type(arg_type_str)
    else:
        assert False, "Unreachable type str"
    return arg_type


def parse_hdc_op(dict_, enable_redperf=True, enable_algchoice=True, redperf_end_only=False):
    hdc_op_name = dict_['intrinsic']
    ID = dict_['number']
    CallingFunction = dict_['parent_func']
    element_type = parse_element_type(dict_['ret_type'])

    argument_types = []
    for arg_type_str in dict_['arg_types']:
        arg_type = parse_type(arg_type_str)
        argument_types.append(arg_type)

    return_type = parse_type(dict_['ret_type'])




    assert hdc_op_name in hdc_op_str_to_enum
    HDCOp = hdc_op_str_to_enum[hdc_op_name]


    op_dict = {
        'hv'  : HDCOperationKind.Vector,
        'hmhv': HDCOperationKind.VectorMatrix,
        'hvhm': HDCOperationKind.VectorMatrix,
        'hmhm': HDCOperationKind.MatrixMatrix,
        'hvhv': HDCOperationKind.VectorVector
    }

    # op_type = op_dict[dict_['op_type']]



    Handle = HDCOperationHandle(CallingFunction, ID, hdc_op_name, dict_['op_type'])
    handle_name = Handle.get_handle_name()


    reducing_dim_extent = get_reducing_dim(HDCOp, return_type, argument_types)


    approximations = []
    if enable_redperf and ApproxReductionPerforation.isApplicableOnHDCOp(HDCOp):

        apx = ApproxReductionPerforation(handle_name, reducing_dim_extent, tune_end_only=redperf_end_only)
        approximations.append(apx)
    if enable_algchoice and ApproxAlgorithmicChoiceHammingDistance.isApplicableOnHDCOp(HDCOp):
        apx = ApproxAlgorithmicChoiceHammingDistance(handle_name)
        approximations.append(apx)
    # TODO: Add other approximations here

    # print("here")
    # print(approximations)

    return ApproxDesc(False, approximations, Handle)











from enum import Enum


class ElementType(Enum):
    INT1 = 1
    INT8 = 2
    INT16 = 3
    INT32 = 4
    INT64 = 5
    FLOAT16 = 6
    FLOAT32 = 7
    FLOAT64 = 8

type_dict = {
    'float' : ElementType.FLOAT32,
    'i1' : ElementType.INT1,
    'i8' : ElementType.INT8,
    'i16' : ElementType.INT16,
    'i32' : ElementType.INT32,
    'i64' : ElementType.INT64,
}

class HDCOperationKind(Enum):
    VectorVector = 1
    VectorMatrix = 2
    MatrixMatrix = 3
    Vector = 4


def isIntegerType(ty : ElementType):
    return ty in [ElementType.INT1, ElementType.INT8,  ElementType.INT16, ElementType.INT32, ElementType.INT64]

def isFloatType(ty : ElementType):
    return ty in [ElementType.FLOAT16, ElementType.FLOAT32,  ElementType.FLOAT64]


class HyperVector:
    def __init__(self, elemTy : ElementType, Dhv : int):
        self.element_type = elemTy
        self.encoding_dimension = Dhv

    def get_encoding_dimension(self) -> int:
        return self.encoding_dimension

    def get_element_type(self) -> ElementType:
        return self.element_type

    def is_integer_type(self) -> bool:
        return isIntegerType(self.element_type)

    def is_float_type(self) -> bool:
        return isFloatType(self.element_type)


class HyperMatrix:
    def __init__(self, elemTy : ElementType, NumRows : int , Dhv : int):
        self.element_type = elemTy
        self.num_rows = NumRows
        self.encoding_dimension = Dhv

    def get_encoding_dimension(self) -> int:
        return self.encoding_dimension

    def get_num_rows(self) -> int:
        return self.num_rows

    def get_element_type(self) -> ElementType:
        return self.element_type

    def is_integer_type(self) -> bool:
        return isIntegerType(self.element_type)

    def is_float_type(self) -> bool:
        return isFloatType(self.element_type)

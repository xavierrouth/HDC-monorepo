# from hdc_approximation_tuner.ApproxTypes import ApproxElementType, ApproxEncodingDimension, ApproxAlgorithmicChoice
# from hdc_approximation_tuner.Types import ElementType
# from hdc_approximation_tuner.HDCPrimitives import HDCOperation
from hdc_approximation_tuner import *

import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result


HammingDist = HDCOperation('HammingDist')
CosineSimilarity = HDCOperation('CosineSimilarity')

apxElemTy = ApproxElementType('ElemTy',ElementType.FLOAT32)
apxDhv = ApproxEncodingDimension('EncDHV',8, 2048, True)
apxAlgChoice = ApproxAlgorithmicChoice('AlgChoice', HammingDist, [HammingDist, CosineSimilarity])


approximations = [apxElemTy, apxDhv, apxAlgChoice]


class TestTuner(MeasurementInterface):

    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        manipulator = ConfigurationManipulator()
        for apx in approximations:
            open_tuner_params = apx.get_opentuner_parameters()
            for params in open_tuner_params:
                manipulator.add_parameter(params)
        return manipulator

    def compile(self, cfg, id):
        """
        Compile a given configuration in parallel
        """
        print(cfg)
        gcc_cmd = 'ls'
        return self.call_program(gcc_cmd)

    def run_precompiled(self, desired_result, input, limit, compile_result, id):
        """
        Run a compile_result from compile() sequentially and return performance
        """
        assert compile_result['returncode'] == 0

        try:
            run_result = self.call_program('./tmp{0}.bin'.format(id))
            assert run_result['returncode'] == 0
        finally:
            self.call_program('rm ./tmp{0}.bin'.format(id))

        return Result(time=6)

    def compile_and_run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        cfg = desired_result.configuration.data
        compile_result = self.compile(cfg, 0)
        return self.run_precompiled(desired_result, input, limit, compile_result, 0)


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    TestTuner.main(argparser.parse_args())

from hdc_approximation_tuner.HDCPrimitives import HDCOperation, hdc_enum_to_op_str
from hdc_approximation_tuner.ApproxDesc import ApproxDesc
from hdc_approximation_tuner.PCMDesc import PCMDesc
from abc import ABC, abstractmethod
from typing import List, Optional


import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result
from opentuner.search.objective import ThresholdAccuracyMinimizeTime
from opentuner.measurement.inputmanager import FixedInputManager


from pyatf.tuning_data import Configuration, CostFunctionError, Cost
from pyatf.search_techniques import AUCBandit
from pyatf.abort_conditions import Evaluations


class OpenTunerHarness(MeasurementInterface):
    def __init__(self, args):
        # breakpoint()
        self.accuracy_target = args.accuracy_target
        self.num_knobs = args.num_knobs
        self.seed_config_path = getattr(args, 'seed_config_path', None)

        objective = ThresholdAccuracyMinimizeTime(self.accuracy_target)
        input_manager = FixedInputManager(size=self.num_knobs)

        super(OpenTunerHarness, self).__init__(args, program_name="hdc_approx", input_manager=input_manager ,objective = objective)
        self.approx_descs = args.approx_descs
        self.evaluate_config_fn = args.evaluate_config_fn

    def manipulator(self):
        manipulator = ConfigurationManipulator()

        for apx_desc in self.approx_descs:
            for params in apx_desc.get_opentuner_parameters():
                manipulator.add_parameter(params)
        return manipulator

    def seed_configurations(self):
        """Load seed configurations from JSON file if provided, otherwise return empty list."""
        if self.seed_config_path:
            try:
                import json
                import os
                if os.path.exists(self.seed_config_path):
                    with open(self.seed_config_path, 'r') as f:
                        config_data = json.load(f)
                        # Handle both single config dict and list of configs
                        if isinstance(config_data, dict):
                            configs = [config_data]
                        elif isinstance(config_data, list):
                            configs = config_data
                        else:
                            print(f"Warning: Invalid format in {self.seed_config_path}, expected dict or list")
                            return []
                        # Reconcile each seed against this manipulator's knob set.
                        # Keys the manipulator doesn't have are dropped; keys it has
                        # but the seed lacks are filled from the manipulator defaults.
                        # Without the fill, OpenTuner KeyErrors while hashing the seed
                        # whenever config_out.json exposes an op the seed predates.
                        manip = self.manipulator()
                        valid_keys = {p.name for p in manip.params}
                        filtered = []
                        for cfg in configs:
                            known = {k: v for k, v in cfg.items() if k in valid_keys}
                            if not known:
                                continue
                            missing = valid_keys - set(known)
                            if missing:
                                print(f"Note: seed missing {len(missing)} knob(s); "
                                      f"using manipulator defaults for: "
                                      f"{', '.join(sorted(missing))}")
                            merged = manip.seed_config()
                            merged.update(known)
                            filtered.append(merged)
                        return filtered
                else:
                    print(f"Warning: Seed config file not found: {self.seed_config_path}")
                    return []
            except Exception as e:
                print(f"Warning: Failed to load seed config from {self.seed_config_path}: {e}")
                return []
        return []

    # HD-HashTable CPU
    # def seed_configurations(self):
    #     return [
    #         { 
    #         'GLOBAL_EncodingDim': 4096, 
    #         'CUSTOM+STRIDE': 1}
    #     ]

    # Clustering GPU - Example hardcoded config
    # def seed_configurations(self):
    #     return [
    #         {
    #         'Z15clustering_nodeIL+llvm.hpvm.hdc.hamming.distance+0+hmhv+RedPerfBase': 0.0, 
    #         'Z15clustering_nodeIL+llvm.hpvm.hdc.hamming.distance+0+hmhv+RedPerfStride': 0.0001, 
    #         'Z15clustering_nodeIL+llvm.hpvm.hdc.hamming.distance+0+hmhv+RedPerfExtent': 1.0, 
    #         'Z15clustering_nodeIL+llvm.hpvm.hdc.hamming.distance+0+hmhv+ALGChoice': 3, 
    #         'GLOBAL_EncodingDim': 1024, 
    #         'CUSTOM+TRAIN_ITERATIONS': 20,
    #         'CUSTOM+DATA_TYPE': 1, 
    #         'CUSTOM+HAMMING_DIST': 0}
    #     ]

    # Classification
    # def seed_configurations(self):
    #  return [
    #     {'Z18training_root_nodeIL+llvm.hpvm.hdc.cossim+2+hmhv+RedPerfBase': 0.0, 
    #     'Z18training_root_nodeIL+llvm.hpvm.hdc.cossim+2+hmhv+RedPerfStride': 0.0001, 
    #     'Z18training_root_nodeIL+llvm.hpvm.hdc.cossim+2+hmhv+RedPerfExtent': 1.0, 
    #     'Z14inference_nodeIL+llvm.hpvm.hdc.cossim+2+hmhv+RedPerfBase': 0.0, 
    #     'Z14inference_nodeIL+llvm.hpvm.hdc.cossim+2+hmhv+RedPerfStride': 0.0001, 
    #     'Z14inference_nodeIL+llvm.hpvm.hdc.cossim+2+hmhv+RedPerfExtent': 1.0, 
    #     'GLOBAL_EncodingDim': 1024, 
    #     'CUSTOM+TRAIN_ITERATIONS': 10}
    #  ]

    # Classification SIM
    # def seed_configurations(self):
    #     return [
    #         {'GLOBAL_PCM_MLC': 1, 
    #         'GLOBAL_EncodingDim': 512, 
    #         'GLOBAL_PCM_ADC_RES': 1, 
    #         'GLOBAL_PCM_ADC_STEP': 1, 
    #         'GLOBAL_PCM_ADC_CEIL': 15, 
    #         'GLOBAL_PCM_WRITE_VERIFY_CYCLE': 1, 
    #         'GLOBAL_PCM_ADC_FLOOR': -15, 
    #         'CUSTOM+TRAIN_ITERATIONS': 20}
    #     ]

    # HD-Graph no-stride
    # def seed_configurations(self):
    #     return [
    #         {'GLOBAL_EncodingDim': 2048}
    #     ]

    def run(self, desired_result, input, limit):
        # breakpoint()
        cfg = desired_result.configuration.data
        run_result = self.evaluate_config_fn(cfg)
        # breakpoint()
        result = Result(time=run_result['time'], accuracy = run_result['accuracy'], energy = run_result['energy'])
        # breakpoint()
        # for k, v in run_result.items():
        #     breakpoint()
        #     if k == "time" or k == "accuracy":
        #         continue
        #     else:
        #         result.set_attribute(k, v)
        return result

    def save_final_config(self, configuration):
        self.manipulator().save_to_file(configuration.data, 'opentuner_final_config.json')


# Helper to convert opentuner config to json readable by HPVM passes 
# for use in eval_config() function
def process_opentuner_config(cfg):

    op_to_redperf = {}
    op_to_alg_choice = {}

    unique_contexts = set()


    hdc_approx_config = []
    for key, value in cfg.items():
        tokens = key.split("+")

        if "GLOBAL" in key:
            # Global approximation, no need to create handle
            if "EncodingDim" in tokens[-1]:
                hdc_config_object = {"EncodingDim" : value}
                hdc_approx_config.append(hdc_config_object)

            if "AutoBinarize" in tokens[-1]:
                hdc_config_object = {"AutoBinarize" : value}
                hdc_approx_config.append(hdc_config_object)

            # Supporting PCM Approximation parameters
            if PCMDesc.is_pcm_param(key):
                hdc_config_object = PCMDesc.generate_hdc_approx_config(key, value)
                hdc_approx_config.append(hdc_config_object)

            continue

        if "CUSTOM" in key:
            # CUSTOM approximation, no need to create handle

            hdc_config_object = {tokens[-1] : value}
            hdc_approx_config.append(hdc_config_object)

            continue

        op_handle = "+".join(tokens[:-1])
        unique_contexts.add(op_handle)

        # Processesing reduction perforation approx type
        if "RedPerf" in tokens[-1]:
            perf_param = tokens[-1]


            if op_handle not in op_to_redperf:
                op_to_redperf[op_handle] = [0,0,0]

            if "Base" in perf_param:
                op_to_redperf[op_handle][0] =  value
            elif "Stride" in perf_param:
                op_to_redperf[op_handle][1] =  value
            elif "Extent" in perf_param:
                op_to_redperf[op_handle][2] =  value
            else:
                assert False, "Unreachable"
        # Processing Algorithmic choice
        if "ALGChoice" in tokens[-1]:
            if op_handle not in op_to_alg_choice:
                op_to_alg_choice[op_handle] = hdc_enum_to_op_str[value]





    for ctx in unique_contexts:
        tokens = ctx.split("+")
        calling_fn = tokens[0]
        op_name = tokens[1]
        number = int(tokens[2])
        op_type = tokens[3]

        output_object = {
            "parent_func": calling_fn,
            "intrinsic": op_name,
            "number": number,
            "op_type": op_type,
            "approximations": [

            ]
        }

        if ctx in op_to_redperf:
            [base, stride, extent] = op_to_redperf[ctx]
            use_float = True
            if use_float:
                num_stride_hops = (1.0 - base) / stride
                subset_hops = extent * num_stride_hops
                end_point = base + (stride * subset_hops)
                red_perf_desc = {"base": base, "stride": stride, "length": end_point}
            else:
                red_perf_desc = {"base": base, "stride": stride, "length": base + (stride * extent)}
            output_object['approximations'].append({"red_perf": red_perf_desc})

        if ctx in op_to_alg_choice:
            output_object['approximations'].append({"alg_choice": op_to_alg_choice[ctx]})

        # Add other approx


        hdc_approx_config.append(output_object)
    return hdc_approx_config









class PYATFCostFunction():
    def __init__(self, execute_fn):
        self.execute_fn = execute_fn



    def __call__(self, configuration: Configuration) -> Cost:
        # add configuration to environment variables
        augmented_env = os.environ.copy()
        for tp_name, tp_value in configuration.items():
            augmented_env[tp_name] = str(tp_value)

        result = self.execute_fn(configuration)
        return float(result['time'])


class PYATFHarness():
    def __init__(self, approx_descs : List[ApproxDesc], evaluate_config_fn, num_evaluations = 100):
        self.num_evaluations = num_evaluations
        self.approx_descs = approx_descs
        self.evaluate_config_fn = evaluate_config_fn
        self.parameters = []
        for apx_desc in self.approx_descs:
            for params in apx_desc.get_pyatf_parameters():
                self.parameters.append(params)

        self.tuner = Tuner().tuning_parameters(*self.parameters).search_technique( AUCBandit() )


    def tune(self):
        cost_fn = PYATFCostFunction(self.evaluate_config_fn)
        tuning_result = self.tuner.tune(cost_fn, Evaluations(evaluations))






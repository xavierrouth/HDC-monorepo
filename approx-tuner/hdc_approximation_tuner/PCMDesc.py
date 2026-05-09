
MLC_PARAM_NAME = "MLC"
ADC_RESOLUTION_NAME = "ADC_RES"
ADC_STEP_NAME = "ADC_STEP"
ADC_FLOOR_NAME = "ADC_FLOOR"
ADC_CEIL_NAME = "ADC_CEIL"
WRITE_VERIFY_CYCLE_NAME = "WRITE_VERIFY_CYCLE"

class PCMDesc:
    MLC_PARAM_NAME = "MLC"
    ADC_RESOLUTION_NAME = "ADC_RES"
    ADC_STEP_NAME = "ADC_STEP"
    ADC_FLOOR_NAME = "ADC_FLOOR"
    ADC_CEIL_NAME = "ADC_CEIL"
    WRITE_VERIFY_CYCLE_NAME = "WRITE_VERIFY_CYCLE"


    def __init__(self, mlc_level_min = 0, mlc_level_max = 4, adc_resolution_min = 0, adc_resolution_max = 8, adc_step_min = 1, adc_step_max = 4, adc_floor_min = 0, adc_floor_max = 4, adc_ceil_min = 0 , adc_ceil_max = 4, write_verify_cycles_min = 0, write_verify_cycles_max = 4):

        self.mlc_level_min = mlc_level_min
        self.mlc_level_max = mlc_level_max

        self.adc_resolution_min = adc_resolution_min
        self.adc_resolution_max = adc_resolution_max

        self.adc_step_min = adc_step_min
        self.adc_step_max = adc_step_max

        self.adc_floor_min = adc_floor_min
        self.adc_floor_max = adc_floor_max

        self.adc_ceil_min = adc_ceil_min
        self.adc_ceil_max = adc_ceil_max

        self.write_verify_cycles_min = write_verify_cycles_min
        self.write_verify_cycles_max = write_verify_cycles_max



    def get_tuning_param_desc(self):

        # Template = ("ParamName", min_val, max_val)
        param_desc = [
            (MLC_PARAM_NAME, self.mlc_level_min, self.mlc_level_max ),
            (ADC_RESOLUTION_NAME, self.adc_resolution_min, self.adc_resolution_max ),
            (ADC_STEP_NAME, self.adc_step_min, self.adc_step_max),
            (ADC_FLOOR_NAME, self.adc_floor_min, self.adc_floor_max),
            (ADC_CEIL_NAME, self.adc_ceil_min, self.adc_ceil_max),
            (WRITE_VERIFY_CYCLE_NAME, self.write_verify_cycles_min, self.write_verify_cycles_max),
        ]

        return param_descs


    @classmethod
    def is_pcm_param(cls, key):
        params_names = [
            MLC_PARAM_NAME ,
            ADC_RESOLUTION_NAME ,
            ADC_STEP_NAME ,
            ADC_FLOOR_NAME ,
            ADC_CEIL_NAME ,
            WRITE_VERIFY_CYCLE_NAME,
        ]

        return any([p in key for p in params_names])

    @classmethod
    def generate_hdc_approx_config(cls, key, value):
        params_names = [
            MLC_PARAM_NAME ,
            ADC_RESOLUTION_NAME ,
            ADC_STEP_NAME ,
            ADC_FLOOR_NAME ,
            ADC_CEIL_NAME ,
            WRITE_VERIFY_CYCLE_NAME,
        ]

        valid_param_name = list(filter(lambda x: x in key, params_names))
        assert len(valid_param_name) == 1, "Must match exactly one parameter"
        return {valid_param_name[0] : value}






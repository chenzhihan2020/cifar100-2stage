from models.mobilenetv2 import mobilenetv2
from FLOPs import get_model_complexity_info


net = mobilenetv2
flops, params = get_model_complexity_info(net,(32,32),as_strings=True,print_per_laryer_stat=True)
print("FLOPs: {}".format(flops))
print("Params: {}" + params)

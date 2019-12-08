from models.preactresnet import preactresnet101
from FLOPs import get_model_complexity_info


net = preactresnet101()
flops, params = get_model_complexity_info(net,(3,32,32),as_strings=True,print_per_layer_stat=False)
print("FLOPs: {}".format(flops))
print("Params: {}".format(params))

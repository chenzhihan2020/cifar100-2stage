from models.squeezenet import squeezenet
from FLOPs import get_model_complexity_info


net = squeezenet()
flops, params = get_model_complexity_info(net,(3,32,32),as_strings=False,print_per_layer_stat=True)
print("FLOPs: {}".format(flops))
print("Params: {}" + params)

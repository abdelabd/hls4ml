from __future__ import print_function
import os

import torch
from hls4ml.converters.pytorch_to_hls import PyTorchModelReader
from hls4ml.utils.config import create_vivado_config
from hls4ml.model.hls_layers import HLSType, IntegerPrecisionType, FixedPrecisionType


class PygModelReader(PyTorchModelReader):
    def __init__(self, config):
        super().__init__(config)
        self.n_node = config['n_node']
        self.n_edge = config['n_edge']
        self.node_dim = config['node_dim']
        self.edge_dim = config['edge_dim']

    def get_weights_data(self, module_name, layer_name, var_name):
        data = None

        # Parameter mapping from pytorch to keras
        torch_paramap = {
            # Conv
            'kernel': 'weight',
            # Batchnorm
            'gamma': 'weight',
            'beta': 'bias',
            'moving_mean': 'running_mean',
            'moving_variance': 'running_var'}

        if var_name not in list(torch_paramap.keys()) + ['weight', 'bias']:
            raise Exception('Pytorch parameter not yet supported!')

        elif var_name in list(torch_paramap.keys()):
            var_name = torch_paramap[var_name]

        try:
            data = self.state_dict[module_name + '.' + layer_name + '.' + var_name].numpy().transpose()
        except KeyError:
            data = self.state_dict[module_name + '.layers.' + layer_name + '.' + var_name].numpy().transpose()

        return data

def pyg_to_hls(model, forward_dict, graph_dims,
               fixed_precision_bits=16,
               fixed_precision_int_bits=6,
               int_precision_bits=16,
               int_precision_signed=False):

    # get graph dimensions
    n = graph_dims.get("n_node_max", 112)
    m = graph_dims.get("n_edge_max", 148)
    p = graph_dims.get("node_dim", 3)
    q = graph_dims.get("edge_dim", 4)
    r = graph_dims.get("relation_dim", q)

    # get precisions
    fp_type = FixedPrecisionType(width=fixed_precision_bits, integer=fixed_precision_int_bits)
    int_type = IntegerPrecisionType(width=int_precision_bits, signed=int_precision_signed)

    # make config
    config = {
        "output_dir": os.getcwd() + "/hls_output",
        "project_name": "myproject",
        "fpga_part": 'xcku115-flvb2104-2-i',
        "clock_period": 5,
        "io_type": "io_parallel",
    }
    config = create_vivado_config(**config)
    config['PytorchModel'] = model
    config['n_node'] = n
    config['n_edge'] = m
    config['node_dim'] = p
    config['edge_dim'] = q
    config['InputShape'] = {
        'NodeAttr': [n, p],
        'EdgeAttr': [m, q],
        'EdgeIndex': [2, m]
    }
    config['InputNodeData'] = 'tb_data/input_node_data.dat'
    config['InputEdgeData'] = 'tb_data/input_edge_data.dat'
    config['InputEdgeIndex'] = 'tb_data/input_edge_index.dat'
    config['OutputPredictions'] = 'tb_data/output_predictions.dat'
    config['HLSConfig']['Model'] = {
        'Precision': 'ap_fixed<16,6>',
        'ReuseFactor': 1,
        'Strategy': 'Latency'
    }

    # make reader
    reader = PygModelReader(config)

    #make layer list
    layer_list = []
    input_shapes = reader.input_shape
    NodeAttr_layer = {
        'name': 'node_attr',
        'class_name': 'InputLayer',
        'input_shape': input_shapes['NodeAttr'],
        'inputs': 'input',
        'dim_names': ['N_NODE', 'NODE_DIM'],
        'precision': fp_type
    }
    layer_list.append(NodeAttr_layer)
    EdgeAttr_layer = {
        'name': 'edge_attr',
        'class_name': 'InputLayer',
        'input_shape': input_shapes['EdgeAttr'],
        'inputs': 'input',
        'dim_names': ['N_EDGE', 'EDGE_DIM'],
        'precision': fp_type
    }
    layer_list.append(EdgeAttr_layer)
    EdgeIndex_layer = {
        'name': 'edge_index',
        'class_name': 'InputLayer',
        'input_shape': input_shapes['EdgeIndex'],
        'inputs': 'input',
        'dim_names': ['TWO', 'N_EDGE'],
        'precision': int_type
    }
    layer_list.append(EdgeIndex_layer)

    # TODO: add aggregation layer if first layer is NodeBlock
    for key, val in forward_dict.items():
        layer_dict = {
            "name": key,
            "class_name": val,
            "n_node": n,
            "n_edge": m,
            "node_dim": p,
            "edge_dim": q,
            "precision": fp_type
        }

        # get n_layers, out_dim
        torch_block = getattr(model, key)
        try:
            torch_layers = torch_block.layers._modules
        except AttributeError:
            torch_layers = torch_block._modules

        lcount = 0
        for lname, l in torch_layers.items():
            if isinstance(l, torch.nn.modules.linear.Linear):
                lcount += 1
                last_layer = l
        layer_dict["n_layers"] = lcount
        layer_dict["out_dim"] = last_layer.out_features

        # get inputs, outputs
        index = len(layer_list)+1
        last_node_update = "node_attr"
        last_edge_update = "edge_attr"
        last_edge_update_aggr = "edge_attr_aggr"
        for l in layer_list:
            if l["class_name"] == "NodeBlock":
                last_node_update = l["outputs"][0]
            elif l["class_name"] == "EdgeBlock":
                last_edge_update = l["outputs"][0]
                last_edge_update_aggr = l["outputs"][1]

        if val=="NodeBlock":
            layer_dict["inputs"] = [last_node_update, last_edge_update_aggr]
            layer_dict["outputs"] = [f"layer{index}_out_P"]
        elif val=="EdgeBlock":
            layer_dict["inputs"] = [last_node_update, last_edge_update, "edge_index"]
            layer_dict["outputs"] = [f"layer{index}_out_L", f"layer{index}_out_Q"]

        layer_list.append(layer_dict)


    return config, reader, layer_list


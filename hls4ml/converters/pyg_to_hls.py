from __future__ import print_function
from hls4ml.converters.pytorch_to_hls import PyTorchModelReader
from hls4ml.utils.config import create_vivado_config
from hls4ml.model.hls_layers import HLSType, IntegerPrecisionType, FixedPrecisionType
import os

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

def pyg_to_hls(model, graph_dims,
               fixed_precision_bits=32,
               fixed_precision_int_bits=16,
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

    R1_layer = {
        'name': 'R1',
        'class_name': 'EdgeBlock',
        'n_node': n,
        'n_edge': m,
        'node_dim': p,
        'edge_dim': q,
        'out_dim': q,
        'inputs': ['node_attr', 'edge_attr', 'edge_index'],
        'outputs': ["layer4_out_L", "layer4_out_Q"],
        'precision': fp_type
    }
    #layer_list.append(R1_layer)

    O_layer = {
        'name': 'O',
        'class_name': 'NodeBlock',
        'n_node': n,
        'n_edge': m,
        'node_dim': p,
        'edge_dim': q,
        'out_dim': p,
        'inputs': ['node_attr', "layer4_out_Q"],
        'outputs': ["layer5_out_P"],
        'precision': fp_type
    }
    #layer_list.append(O_layer)

    R2_layer = {
        'name': 'R2',
        'class_name': 'EdgeBlock',
        'n_node': n,
        'n_edge': m,
        'node_dim': p,
        'edge_dim': q,
        'out_dim': 1,
        'inputs': ['layer5_out_P', 'layer4_out_L', 'edge_index'],
        'outputs': ['layer6_out_L', 'layer6_out_Q'],
        'precision': fp_type
    }
    #layer_list.append(R2_layer)
    block_layers = [R1_layer, O_layer, R2_layer]
    for l in block_layers: layer_list.append(l)

    return config, reader, layer_list


from __future__ import print_function
from hls4ml.converters.pytorch_to_hls import PyTorchModelReader

class PygModelReader(PyTorchModelReader):

    def __init__(self, config):
        super().__init__(config)
        self.n_node = config['n_node']
        self.n_edge = config['n_edge']
        self.node_dim = config['node_dim']
        self.edge_dim= config['edge_dim']

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

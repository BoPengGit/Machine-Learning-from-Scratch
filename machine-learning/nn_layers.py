class Layer(object):
    """Neural Network Layers"""

    def __init__(self):
        pass

    def fully_connected(self, num_nodes):
        # Append fully_connected layer to input architecture.
        num_last_layer_nodes = self.architecture.layer[-1].num_units
        self.architecture = self.architecture.add(layer)

    def batch_normalization(self):
        pass

    def dropout(self):
        pass

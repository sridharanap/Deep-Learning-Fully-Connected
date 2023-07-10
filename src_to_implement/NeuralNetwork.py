import copy


class NeuralNetwork:
    def __init__(self, optimizer: object):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None

    def append_layer(self, layer):
        if layer.trainable is True:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def forward(self):
        input_tensor,label_tensor=self.data_layer.next()
        self.label_tensor=label_tensor
        for j in self.layers:
            input_tensor = j.forward(input_tensor)
        output_loss = self.loss_layer.forward(input_tensor,label_tensor)
        return output_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for j in reversed(self.layers):
            error_tensor = j.backward(error_tensor)
        return error_tensor

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for j in self.layers:
            input_tensor = j.forward(input_tensor)
        return input_tensor

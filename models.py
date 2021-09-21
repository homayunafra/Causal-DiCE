import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, params):
        super().__init__()
        """Initialize parameters
        :params input_size: number of units in the input layer.
        :params output_size: number of output units.
        :params hidden_cnt: number of hidden layers.
        :params hidden_size: vector containing sizes of hidden layers or a scalar for a single hidden layer network.
        """

        self.inputSize = params['input_size']
        self.outputSize = params['output_size']
        self.hiddenCnt = params['hidden_cnt']
        self.hiddenSizes = params['hidden_size']

        ########## initialize weight values ##########
        # Currently works only for single hidden layer network
        if self.hiddenCnt == 1:
            self.hidden = nn.Linear(self.inputSize, self.hiddenSizes, bias=False) # init of the hidden layer
            self.output = nn.Linear(self.hiddenSizes, self.outputSize, bias=False) # init of the output layer

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.1, inplace=False)
            self.batchnorm1 = nn.BatchNorm1d(self.hiddenSizes)
        else:
            for hidden_index in range(0, self.hiddenCnt-1):
                if hidden_index == 0:
                    self.hidden[hidden_index] = nn.Linear(self.inputSize, self.hiddenSizes[hidden_index])
                else:
                    self.hidden[hidden_index] = nn.Linear(self.hiddenSizes[hidden_index-1], self.hiddenSizes[hidden_index])
            self.output = nn.Linear(self.hiddenSizes[self.hiddenCnt - 1], self.outputSize)

    def forward(self, X):
        x = self.relu(self.hidden(X))
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

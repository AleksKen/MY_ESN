import torch
import torch.nn as nn

# Определение резервуарной сети (Echo State Network) без наблюдателя
class EchoStateNetworkWithoutObserver(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size):
        super(EchoStateNetworkWithoutObserver, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size

        self.W_in = nn.Parameter(torch.randn(reservoir_size, input_size))
        self.W_res = nn.Parameter(torch.randn(reservoir_size, reservoir_size))

        self.reservoir = torch.zeros(1, reservoir_size)

    def forward(self, input_data):
        input_data = input_data.view(-1, self.input_size)
        input_data = torch.tanh(torch.mm(input_data, self.W_in.t()))

        reservoir_input = torch.clone(self.reservoir)
        reservoir_input = torch.tanh(torch.mm(reservoir_input, self.W_res.t()) + input_data)

        return reservoir_input



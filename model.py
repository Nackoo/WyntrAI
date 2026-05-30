import torch
import torch.nn as nn


class ChatNet(nn.Module):
    """
    A simple 3-layer feedforward neural network for intent classification.
    Input:  bag-of-words vector (size = vocabulary length)
    Output: probability distribution over known response tags
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(ChatNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)

    def get_layer_activations(self, x):
        """
        Returns intermediate activations for visualization purposes.
        Used by the neural network visualizer (index.html).
        """
        activations = {}
        activations["input"] = x.detach().numpy().tolist()

        out = self.network[0](x)   # Linear 1
        activations["linear1"] = out.detach().numpy().tolist()

        out = self.network[1](out) # ReLU 1
        activations["relu1"] = out.detach().numpy().tolist()

        out = self.network[3](out) # Linear 2  (skip Dropout index 2)
        activations["linear2"] = out.detach().numpy().tolist()

        out = self.network[4](out) # ReLU 2
        activations["relu2"] = out.detach().numpy().tolist()

        out = self.network[6](out) # Linear 3  (skip Dropout index 5)
        activations["output_logits"] = out.detach().numpy().tolist()

        probs = torch.softmax(out, dim=0)
        activations["output_probs"] = probs.detach().numpy().tolist()

        return activations
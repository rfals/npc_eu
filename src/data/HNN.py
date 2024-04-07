import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from functools import reduce

# create HNN model class
class HNN(nn.Module):

    def __init__(self, nn_hyps):

        super(HNN, self).__init__()

        n_features = nn_hyps['n_features']
        nodes = nn_hyps['nodes']
        x_pos = nn_hyps['x_pos']
        dropout_rate = nn_hyps['dropout_rate']
        add_trends_to = nn_hyps['add_trends_to']

        n_group = len(x_pos)  # - len(add_trends_to)
        n_by_group = []
        x_indices = []

        for i in range(n_group):

            if i > (n_group - len(add_trends_to) - 1):
                x_indices.append(torch.tensor(x_pos[i]))
                n_by_group.append(1)
            else:
                x_indices.append(torch.tensor(x_pos[i]))
                n_by_group.append(torch.tensor(math.sqrt(len(x_pos[i]))))

        # Input layer
        self.input_layer = torch.nn.ModuleList(
            nn.Linear(n_features[i], nodes[i][0]) for i in range(n_group))

        # Hidden layers
        self.first_layer = []
        self.hidden_layers = []
        self.output_layer = []

        for i in range(n_group):

            self.first_layer.append(nn.Linear(nodes[i][0], nodes[i][0]))
            self.hidden_layers.append(nn.ModuleList(
                nn.Linear(nodes[i][j], nodes[i][j+1]) for j in range(0, (len(nodes[i])-1))))

            # Output layer
            self.output_layer.append(nn.Linear(nodes[i][len(nodes[i])-1], 1))

        self.n_by_group = n_by_group
        self.n_group = n_group
        self.n_features = n_features
        self.n_layer = len(nodes[0])
        self.x_indices = x_indices
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)
        self.add_trends_to = add_trends_to
        self.n_add_trends_to = len(add_trends_to)
        self.relu = nn.ReLU()

    def forward(self, x):

        # Set up
        dat = []  # data by group
        xx = [None]*self.n_group  # create list of tensors

        for i in range(self.n_group):
            dat.append(x[:, self.x_indices[i]])
            #dat[i] = torch.div(dat[i],self.n_by_group[i])
            dat[i] = dat[i]/self.n_by_group[i]

        
        # Input layer
        for i in range(self.n_group):
            # xx.append(self.input_layer[i](dat[i]))
            xx[i] = self.input_layer[i](dat[i])
            xx[i] = self.relu(xx[i])
            # xx[i] = F.dropout(xx[i], self.dropout_rate)

        # Hidden layers
        for i in range(self.n_group):
            xx[i] = self.first_layer[i](xx[i])
            xx[i] = self.relu(xx[i])
            xx[i] = self.dropout(xx[i])

        for i in range(self.n_group):
            for j in range(self.n_layer-1):
                xx[i] = self.hidden_layers[i][j](xx[i])
                xx[i] = self.relu(xx[i])
                xx[i] = self.dropout(xx[i])

        # Output layer
        for i in range(self.n_group):
            xx[i] = self.output_layer[i](xx[i])

        # Get the trends (coefficients)
        trends = []
        alt = list(range((self.n_group-self.n_add_trends_to), (self.n_group)))
        trends = [xx[i] for i in alt]

        # Get the output layer predictions
        alt = list(range(0, (self.n_group-self.n_add_trends_to)))
        xx = [xx[i] for i in alt]

        # Trends identification
        for i in range(self.n_add_trends_to):
            trends[i] = torch.abs(trends[i])

        # Multiply trends and gaps to give the contribution of each hemisphere
        gaps_alt = [i.clone() for i in xx]
        for i in range(self.n_add_trends_to):
            xx[i] = torch.mul(xx[i], trends[i])

        gaps = gaps_alt[0]
        components = xx[0]
        for i in range(1, (self.n_group-self.n_add_trends_to)):
            components = torch.hstack([components, xx[i]])
            gaps = torch.hstack([gaps, gaps_alt[i]])

        alt = trends[0]
        for i in range(1, self.n_add_trends_to):
            alt = torch.hstack([alt, trends[i]])
        trends = alt

        yhat = torch.sum(components, dim=1)

        # Results
        results = [yhat, components, trends, gaps]
        return results
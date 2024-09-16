import torch.nn.functional as F
import torch
import torch.nn as nn
from model.utils import fix_seed

fix_seed()


class SugaiNetArchitecture(nn.Module):
    """ LSTM Network-Based Estimation of Ground Reaction Forces During Walking
     in Stroke Patients Using Markerless Motion Capture System """
    def __init__(self, repr_dim, opt):
        super(SugaiNetArchitecture, self).__init__()
        self.input_dim = len(opt.kinematic_diffusion_col_loc)
        self.output_dim = repr_dim - self.input_dim
        self.input_col_loc = opt.kinematic_diffusion_col_loc
        self.output_col_loc = [i for i in range(repr_dim) if i not in self.input_col_loc]
        self.device = 'cuda'

        self.hidden1 = nn.LSTM(self.input_dim, 216, 1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.hidden2 = nn.LSTM(216, 76, 1, batch_first=True)
        self.dropout2 = nn.Dropout(0.25)
        self.output = nn.Linear(76, self.output_dim, bias=True)

    def end_to_end_prediction(self, x):
        input = x[0][:, :, self.input_col_loc]
        x, _ = self.hidden1(input)
        x = self.dropout1(x)
        x, _ = self.hidden2(x)
        x = self.dropout2(x)
        output_pred = self.output(x)
        return output_pred

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=5e-3)

    def loss_fun(self, output_pred, output_true):
        return F.l1_loss(output_pred, output_true, reduction='none')

    def __str__(self):
        return 'sugainet'


class Transpose(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self._dim1, self._dim2 = dim1, dim2

    def extra_repr(self):
        return "{}, {}".format(self._dim1, self._dim2)

    def forward(self, input):
        return input.transpose(self._dim1, self._dim2)


class GroundLinkArchitecture(nn.Module):
    def __init__(self, repr_dim, opt):
        super().__init__()
        self.input_dim = len(opt.kinematic_diffusion_col_loc)
        self.output_dim = repr_dim - self.input_dim
        self.input_col_loc = opt.kinematic_diffusion_col_loc
        self.output_col_loc = [i for i in range(repr_dim) if i not in self.input_col_loc]
        self.device = 'cuda'

        cnn_kernel = 7
        cnn_dropout = 0.0
        fc_depth = 3
        fc_dropout = 0.2
        cnn_features = [self.input_dim, 128, 128, 256, 256]
        features_out = self.output_dim
        ## Preprocess part
        pre_layers = [  # N x F x J x [...]
            torch.nn.Flatten(start_dim=2, end_dim=-1),  # N x F x C
            Transpose(-2, -1),  # N x C x F
        ]
        ## Convolutional part
        conv = lambda c_in, c_out: torch.nn.Conv1d(c_in, c_out, cnn_kernel, padding=cnn_kernel // 2,
                                                   padding_mode="replicate")
        cnn_layers = []
        for c_in, c_out in zip(cnn_features[:-1], cnn_features[1:]):  # N x C x F
            cnn_layers += [
                torch.nn.Dropout(p=cnn_dropout),  # N x Ci x F
                conv(c_in, c_out),  # N x Ci x F
                torch.nn.ELU(),  # N x Ci x F
            ]
        ## Fully connected part
        fc_layers = [Transpose(-2, -1)]  # N x F x Cn
        for _ in range(fc_depth - 1):
            fc_layers += [  # N x F x Ci
                torch.nn.Dropout(p=fc_dropout),  # N x F x Ci
                torch.nn.Linear(cnn_features[-1], cnn_features[-1]),  # N x F x Ci
                torch.nn.ELU()  # N x F x Ci
            ]
        fc_layers += [  # N x F x Ci
            torch.nn.Dropout(p=fc_dropout),  # N x F x 2*Co
            torch.nn.Linear(cnn_features[-1], features_out, bias=False),  # N x F x Co
            # torch.nn.Unflatten(-1, (2, features_out)),  # N x F x 2 x Co
            # torch.nn.Softplus(),  # N x F x 2 x Co
        ]
        # commented Softplus because they were only predicting positive vGRF, whereas we have normalized 3D GRF.
        layers = pre_layers + cnn_layers + fc_layers
        layers = self.initialize(layers)
        self.pre_layers = torch.nn.Sequential(*layers[:len(pre_layers)])
        self.cnn_layers = torch.nn.Sequential(*layers[len(pre_layers):len(pre_layers) + len(cnn_layers)])
        self.fc_layers = torch.nn.Sequential(*layers[len(pre_layers) + len(cnn_layers):])

    def end_to_end_prediction(self, x):
        input = x[0][:, :, self.input_col_loc]
        sequence = self.pre_layers(input)
        sequence = self.cnn_layers(sequence)
        output_pred = self.fc_layers(sequence)
        return output_pred

    def get_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)

    def loss_fun(self, output_pred, output_true):
        return F.mse_loss(output_pred, output_true, reduction='none')

    def __str__(self):
        return 'groundlink'

    @staticmethod
    def initialize(layers):
        GAINS = {
            torch.nn.Sigmoid: torch.nn.init.calculate_gain("sigmoid"),
            torch.nn.ReLU: torch.nn.init.calculate_gain("relu"),
            torch.nn.LeakyReLU: torch.nn.init.calculate_gain("leaky_relu"),
            torch.nn.ELU: torch.nn.init.calculate_gain("relu"),
            torch.nn.Softplus: torch.nn.init.calculate_gain("relu"),
        }
        for layer, activation in zip(layers[:-1], layers[1:]):
            if len(list(layer.parameters())) > 0 and type(activation) in GAINS:
                if not isinstance(activation, type):
                    activation = type(activation)
                if activation not in GAINS:
                    raise Exception("Initialization not defined for activation '{}'.".format(type(activation)))
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(layer.weight, GAINS[activation])
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
                elif isinstance(layer, torch.nn.Conv1d):
                    torch.nn.init.xavier_normal_(layer.weight, GAINS[activation])
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
                else:
                    raise Exception("Initialization not defined for layer '{}'.".format(type(layer)))
        return layers




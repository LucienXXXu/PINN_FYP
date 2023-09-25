import torch
import torch.nn as nn
from warnings import warn


class FCNN(nn.Module):
    """A fully connected neural network.

    :param n_input_units: Number of units in the input layer, defaults to 1.
    :type n_input_units: int
    :param n_output_units: Number of units in the output layer, defaults to 1.
    :type n_output_units: int
    :param n_hidden_units: [DEPRECATED] Number of hidden units in each layer
    :type n_hidden_units: int
    :param n_hidden_layers: [DEPRECATED] Number of hidden mappsings (1 larger than the actual number of hidden layers)
    :type n_hidden_layers: int
    :param actv: The activation layer constructor after each hidden layer, defaults to `torch.nn.Tanh`.
    :type actv: class
    :param hidden_units: Number of hidden units in each hidden layer. Defaults to (32, 32).
    :type hidden_units: Tuple[int]

    .. note::
        The arguments "n_hidden_units" and "n_hidden_layers" are deprecated in favor of "hidden_units".
    """

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None, n_hidden_layers=None,
                 actv=nn.Tanh, hidden_units=None):
        r"""Initializer method.
        """
        super(FCNN, self).__init__()

        # FORWARD COMPATIBILITY
        # If only one of {n_hidden_unit, n_hidden_layers} is specified, fill-in the other one
        if n_hidden_units is None and n_hidden_layers is not None:
            n_hidden_units = 32
        elif n_hidden_units is not None and n_hidden_layers is None:
            n_hidden_layers = 1

        # FORWARD COMPATIBILITY
        # When {n_hidden_unit, n_hidden_layers} are specified, construct an equivalent hidden_units if None is provided
        if n_hidden_units is not None or n_hidden_layers is not None:
            if hidden_units is None:
                hidden_units = tuple(n_hidden_units for _ in range(n_hidden_layers + 1))
                warn(f"`n_hidden_units` and `n_hidden_layers` are deprecated, "
                     f"pass `hidden_units={hidden_units}` instead",
                     FutureWarning)
            else:
                warn(f"Ignoring `n_hidden_units` and `n_hidden_layers` in favor of `hidden_units={hidden_units}`",
                     FutureWarning)

        # If none of {n_hidden_units, n_hidden_layers, hidden_layers} are specified, use (32, 32) by default
        if hidden_units is None:
            hidden_units = (32, 32)

        # If user passed in a list, iterator, etc., convert it to tuple
        if not isinstance(hidden_units, tuple):
            hidden_units = tuple(hidden_units)

        units = (n_input_units,) + hidden_units
        layers = []
        for i in range(len(units) - 1):
            layers.append(nn.Linear(units[i], units[i + 1]))
            layers.append(actv())
        # There's not activation in after the last layer
        layers.append(nn.Linear(units[-1], n_output_units))
        self.NN = torch.nn.Sequential(*layers)

    def forward(self, t):
        x = self.NN(t)
        return x


class Resnet(nn.Module):
    """A residual network with a trainable linear skip connection between input and output

    :param n_input_units: Number of units in the input layer, defaults to 1.
    :type n_input_units: int
    :param n_input_units: Number of units in the output layer, defaults to 1.
    :type n_input_units: int
    :param n_hidden_units: [DEPRECATED] Number of hidden units in each layer
    :type n_hidden_units: int
    :param n_hidden_layers: [DEPRECATED] Number of hidden mappsings (1 larger than the actual number of hidden layers)
    :type n_hidden_layers: int
    :param actv: the activation layer constructor after each hidden layer, defaults to `torch.nn.Tanh`.
    :type actv: class
    :param hidden_units: Number of hidden units in each hidden layer. Defaults to (32, 32).
    :param hidden_units: Tuple[int]
    """

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None, n_hidden_layers=None, actv=nn.Tanh,
                 hidden_units=(32, 32)):
        super(Resnet, self).__init__()

        self.residual = FCNN(
            n_input_units=n_input_units,
            n_output_units=n_output_units,
            n_hidden_units=n_hidden_units,
            n_hidden_layers=n_hidden_layers,
            actv=actv,
            hidden_units=hidden_units,
        )
        self.skip_connection = nn.Linear(n_input_units, n_output_units, bias=False)

    def forward(self, t):
        x = self.skip_connection(t) + self.residual(t)
        return x


class DenseResNet(nn.Module):
    """
    This is a ResNet Class.
    -> dim_in: network's input dimension
    -> dim_out: network's output dimension
    -> num_resnet_blocks: number of ResNet blocks
    -> num_layers_per_block: number of layers per ResNet block
    -> num_neurons: number of neurons in each layer
    -> activation: Non-linear activations function that you want to use. E.g. nn.Sigmoid(), nn.ReLU()
    -> fourier_features: whether to pass the inputs through Fourier mapping. E.g. True or False
    -> m_freq: how many frequencies do you want the inputs to be mapped to
    -> sigma: controls the spectrum of frequencies. 
              If sigma is greater more frequencies are consider. 
              You can also look at it as sampling from the standard normal, Z~N(0, 1), 
              and mapping to another normal, X~N(\mu, \sigma^2), using x = mu + sigma*z.
    -> tune_beta: do you want to consider the parameter beta in the activation functions in each layer? E.g., Tanh(beta*x).
                  In practice it is observed that training beta (i.e. tune_beta=True) could improve convergence. 
                  If tune_beta=False, you get the a fixed beta i.e. beta=1.
    -> The method model_capacity() returns the number of layers and parameters in the network.
    """
    def __init__(self, dim_in=2, dim_out=1, num_resnet_blocks=3, 
                 num_layers_per_block=2, num_neurons=50, activation=nn.Sigmoid(), 
                 fourier_features=False, m_freqs=100, sigma=10, tune_beta=False):
        super(DenseResNet, self).__init__()

        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.fourier_features = fourier_features
        self.activation = activation
        self.tune_beta = tune_beta

        if tune_beta:
            self.beta0 = nn.Parameter(torch.ones(1, 1))
            self.beta = nn.Parameter(torch.ones(self.num_resnet_blocks, self.num_layers_per_block))

        else: 
            self.beta0 = torch.ones(1, 1)
            self.beta = torch.ones(self.num_resnet_blocks, self.num_layers_per_block)

        self.first = nn.Linear(dim_in, num_neurons)

        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(num_neurons, num_neurons) 
                for _ in range(num_layers_per_block)]) 
            for _ in range(num_resnet_blocks)])

        self.last = nn.Linear(num_neurons, dim_out)

        if fourier_features:
            self.first = nn.Linear(2*m_freqs, num_neurons)
            self.B = nn.Parameter(sigma*torch.randn(dim_in, m_freqs)) # to converts inputs to m_freqs

    def forward(self, x):
        if self.fourier_features:
            cosx = torch.cos(torch.matmul(x, self.B))
            sinx = torch.sin(torch.matmul(x, self.B))
            x = torch.cat((cosx, sinx), dim=1)
            x = self.activation(self.beta0*self.first(x)) 

        else:
            x = self.activation(self.beta0*self.first(x))

        for i in range(self.num_resnet_blocks):
            z = self.activation(self.beta[i][0]*self.resblocks[i][0](x))

            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.beta[i][j]*self.resblocks[i][j](z))

            x = z + x

        out = self.last(x)

        return out

    def model_capacity(self):
        """
        Prints the number of parameters and the number of layers in the network
        """
        number_of_learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_layers = len(list(self.parameters()))
        print("\n\nThe number of layers in the model: %d" % num_layers)
        print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)  
    
    
    
class MonomialNN(nn.Module):
    """A network that expands its input to a given list of monomials.

    Its output shape will be (n_samples, n_input_units * n_degrees)

    :param degrees: max degree to be included, or a list of degrees that will be used
    :type degrees: int or list[int] or tuple[int]
    """

    def __init__(self, degrees):
        super(MonomialNN, self).__init__()

        if isinstance(degrees, int):
            degrees = [d for d in range(1, degrees + 1)]
        self.degrees = tuple(degrees)

        if len(self.degrees) == 0:
            raise ValueError(f"No degrees used, check `degrees` argument again")
        if 0 in degrees:
            warn("One of the degrees is 0 which might introduce redundant features")
        if len(set(self.degrees)) < len(self.degrees):
            warn(f"Duplicate degrees found: {self.degrees}")

    def forward(self, x):
        return torch.cat([x ** d for d in self.degrees], dim=1)

    def __repr__(self):
        return f"{self.__class__.__name__}(degrees={self.degrees})"

    def __str__(self):
        return self.__repr__()


class SinActv(nn.Module):
    """The sin activation function.
    """

    def __init__(self):
        """Initializer method.
        """
        super().__init__()

    def forward(self, input_):
        return torch.sin(input_)



class Swish(nn.Module):
    r"""The swish activation function: :math:`\mathrm{swish}(x)=x\sigma(\beta x)=\frac{x}{1+e^{-\beta x}}`.

    :param beta: The :math:`\beta` parameter in the swish activation.
    :type beta: float
    :param trainable: Whether scalar :math:`\beta` can be trained
    :type trainable: bool
    """

    def __init__(self, beta=1.0, trainable=False):
        super(Swish, self).__init__()

        beta = float(beta)
        self.trainable = trainable
        if trainable:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

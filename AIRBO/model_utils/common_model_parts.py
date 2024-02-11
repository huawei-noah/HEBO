import torch
import torch.nn as nn
from torch.nn import functional as F
from collections.abc import Iterable
import math


def compute_conv2d_output_shape(h_in, w_in, kernel_size, padding, stride, dilation=1):
    """
    compute the output shape of the conv2D operation
    """
    if isinstance(kernel_size, Iterable) is False:
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, Iterable) is False:
        padding = (padding, padding)
    if isinstance(stride, Iterable) is False:
        stride = (stride, stride)
    if isinstance(dilation, Iterable) is False:
        dilation = (dilation, dilation)

    h_out = int(math.floor(
        (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    ))

    w_out = int(math.floor(
        (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    ))

    return (h_out, w_out)


def compute_conv2dTranspose_output_shape(h_in, w_in, kernel_size, padding, stride,
                                         output_padding=0, dilation=1):
    """
    compute the output shape of conv2DTranspose operation
    """
    if isinstance(kernel_size, Iterable) is False:
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, Iterable) is False:
        padding = (padding, padding)
    if isinstance(stride, Iterable) is False:
        stride = (stride, stride)
    if isinstance(dilation, Iterable) is False:
        dilation = (dilation, dilation)
    if isinstance(output_padding, Iterable) is False:
        output_padding = (output_padding, output_padding)

    h_out = (h_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) \
            + output_padding[0] + 1

    w_out = (w_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) \
            + output_padding[1] + 1

    return (h_out, w_out)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="leaky_relu",
                 act_in_last_layer=False, skip_connection=False, norm_method=None):
        """
        A multi-layer perceptron
        :param input_dim: input dimension
        :param hidden_dims: a list of hidden dims
        :param output_dim: output dimension
        :param activation: which activation to use, can be "relu", "tanh", "sigmoid", "leaky_relu"
        :param act_in_last_layer: whether to use activation in the last layer
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.layer_dims = [input_dim, *hidden_dims, output_dim]
        self.layer_num = len(self.layer_dims) - 1
        self.act_in_last_layer = act_in_last_layer
        self.skip_connection = skip_connection
        self.norm_method = norm_method

        self.layers = []
        for i in range(self.layer_num):
            linear_layer = nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
            if self.norm_method is not None and i != self.layer_num - 1:  # no need bn at the last layer:
                if self.norm_method == 'spectral_norm':
                    sn_l = nn.utils.spectral_norm(linear_layer)
                    self.layers.append(sn_l)
                elif self.norm_method == 'batch_norm':
                    bn = nn.BatchNorm1d(self.layer_dims[i+1])
                    self.layers.append(linear_layer)
                    self.layers.append(bn)
                else:
                    raise ValueError("Invalid norm_method:", self.norm_method)
            else:
                self.layers.append(linear_layer)

            if i != self.layer_num - 1 or self.act_in_last_layer:
                if activation == "relu":
                    self.layers.append(torch.nn.ReLU())
                elif activation == "tanh":
                    self.layers.append(torch.nn.Tanh())
                elif activation == "sigmoid":
                    self.layers.append(torch.nn.Sigmoid())
                elif activation == "leaky_relu":
                    self.layers.append(torch.nn.LeakyReLU())
                else:
                    raise ValueError("Unsupported activation type: ", activation)
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        if self.skip_connection:
            o = x + self.model(x)
        else:
            o = self.model(x)
        return o


class CopyModule(nn.Module):
    def __init__(self):
        """
        A layer doing nothing but return the inputs
        """
        super(CopyModule, self).__init__()

    def forward(self, x):
        return x


class MonoNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, group_dims):
        super(MonoNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.group_dims = group_dims
        self.n_group_dim = sum(group_dims)
        self.model = MLP(self.input_dim, self.hidden_dims, self.n_group_dim)

    def forward(self, x):
        h = self.model(x)
        group_outputs = []
        d_ind = 0
        for pgd in self.group_dims:
            pgh = h[:, d_ind:d_ind + pgd]
            pgo, _ = torch.max(pgh, dim=1, keepdim=True)
            group_outputs.append(pgo)
            d_ind += pgd
        output, _ = torch.min(
            torch.concat(group_outputs, dim=1),
            dim=1, keepdim=True
        )
        return output

    def __call__(self, *args, **kwargs):
        for module in self.model.modules():
            if hasattr(module, 'weight'):
                w = module.weight.data
                w = w.clamp(0.0, None)
                module.weight.data = w
        return self.forward(*args, **kwargs)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1,
                 deconv=False, output_padding=0,
                 use_batch_norm=True, activation="leaky_relu"):
        """
        A single conv2d/deconv2d layer + batch norm + activation
        :param in_channels: input channel num
        :param out_channels: output channel num
        :param kernel_size: kernel size
        :param stride: stride
        :param padding: padding
        :param dilation: dilation
        :param deconv: whether to use DeconvTranspose2D
        :param output_padding: only work if deconv is true
        :param use_batch_norm: whether to use batch normalization
        :param activation: which activation to use
        """
        super(ConvLayer, self).__init__()

        layer_list = []
        conv_filter = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation) if deconv is False \
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                    stride, padding, output_padding)
        layer_list.append(conv_filter)

        if use_batch_norm:
            layer_list.append(nn.BatchNorm2d(num_features=out_channels))

        act = None
        if activation == "leaky_relu":
            act = nn.LeakyReLU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation: {}".format(activation))

        if act is not None:
            layer_list.append(act)

        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)


class MultiConvLayers(nn.Module):
    def __init__(self, layer_configurations):
        """
        Multiple conv2d/deconv2d layers
        :param layer_configurations: a list of dict, each of which has key list= [in_channels,
                                out_channels, kernel_size, stride, padding, dilation, is_deconv,
                                output_padding, use_batch_norm, activation]
        """
        super(MultiConvLayers, self).__init__()
        self.layer_configurations = layer_configurations
        layer_list = []
        for i in range(len(layer_configurations)):
            if i > 0 and layer_configurations[i]["in_channels"] \
                    != layer_configurations[i - 1]["out_channels"]:
                raise ValueError("Unmatched in_channels with previous layer's out_channels")

            cfg = layer_configurations[i]
            in_channels = cfg["in_channels"]
            out_channels = cfg["out_channels"]
            kernel_size = cfg["kernel_size"]
            stride = cfg.get("stride", 1)
            padding = cfg.get("padding", 0)
            dilation = cfg.get("dilation", 1)
            is_deconv = cfg.get("is_deconv", False)
            output_padding = cfg.get("output_padding", None)
            use_batch_norm = cfg.get("use_batch_norm", True)
            activation = cfg.get("activation", "relu")

            layer_list.append(ConvLayer(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, is_deconv, output_padding,
                                        use_batch_norm, activation))

        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, sample_shape, latent_dim, conv_filter_cfgs,
                 enc_fc_hidden_dims=(64, 32), enc_fc_out_dim=32,
                 enc_mu_hdims=(16,), enc_logvar_hdims=(16,)):
        super(Encoder, self).__init__()
        self.sample_shape = sample_shape
        self.latent_dim = latent_dim
        self.conv_filter_cfgs = conv_filter_cfgs

        self.shape_changes = [i for i in range(len(self.conv_filter_cfgs))
                              if self.conv_filter_cfgs[i]["stride"] > 1]

        # encoder
        self.enc_conv_model = MultiConvLayers(conv_filter_cfgs) if len(
            conv_filter_cfgs) > 0 else None
        self.enc_flatten = nn.Flatten()
        self.latent_shape = sample_shape // (2 ** len(self.shape_changes))
        self.enc_fc_input_dim = self.latent_shape * self.latent_shape \
                                * (conv_filter_cfgs[-1]["out_channels"]
                                   if len(conv_filter_cfgs) > 0 else 1)
        self.enc_fc_model = MLP(self.enc_fc_input_dim, enc_fc_hidden_dims, enc_fc_out_dim)
        self.enc_mu_model = MLP(enc_fc_out_dim, enc_mu_hdims, latent_dim)
        self.enc_logvar_model = MLP(enc_fc_out_dim, enc_logvar_hdims, latent_dim)

    def forward(self, x):
        h = x
        if self.enc_conv_model is not None:
            h = self.enc_conv_model(h)
        h = self.enc_flatten(h)
        h = F.relu(self.enc_fc_model(h))
        return self.enc_mu_model(h), self.enc_logvar_model(h)


class EncoderWithY(nn.Module):
    def __init__(self, sample_shape, latent_dim, y_dim, conv_filter_cfgs,
                 enc_fc_hidden_dims=(64, 32), enc_fc_out_dim=32,
                 enc_mu_hdims=(16,), enc_logvar_hdims=(16,)):
        super(EncoderWithY, self).__init__()
        self.sample_shape = sample_shape
        self.latent_dim = latent_dim
        self.conv_filter_cfgs = conv_filter_cfgs
        self.y_dim = y_dim

        self.shape_changes = [i for i in range(len(self.conv_filter_cfgs))
                              if self.conv_filter_cfgs[i]["stride"] > 1]

        # encoder
        self.enc_conv_model = MultiConvLayers(conv_filter_cfgs) \
            if len(conv_filter_cfgs) > 0 else None
        self.enc_flatten = nn.Flatten()
        self.latent_shape = (sample_shape[0] // (2 ** len(self.shape_changes)),
                             sample_shape[1] // (2 ** len(self.shape_changes)))
        self.enc_fc_input_dim = self.latent_shape[0] * self.latent_shape[1] \
                                * (conv_filter_cfgs[-1]["out_channels"]
                                   if len(conv_filter_cfgs) > 0 else 1)
        self.enc_fc_model = MLP(self.enc_fc_input_dim, enc_fc_hidden_dims, enc_fc_out_dim)
        self.enc_mu_model = MLP(enc_fc_out_dim, enc_mu_hdims, latent_dim)
        self.enc_logstd_model = MLP(enc_fc_out_dim, enc_logvar_hdims, latent_dim)
        # self.enc_mu_model = MLP(enc_fc_out_dim+y_dim, enc_mu_hdims, latent_dim)
        # self.enc_logvar_model = MLP(enc_fc_out_dim+y_dim, enc_logvar_hdims, latent_dim)

    def forward(self, x, y):
        h = x
        if self.enc_conv_model is not None:
            h = self.enc_conv_model(h)
        h = self.enc_flatten(h)
        h = F.leaky_relu(self.enc_fc_model(h))
        # mu = self.enc_mu_model(torch.cat([h, y], dim=1))
        mu = self.enc_mu_model(h)
        # logvar = self.enc_logvar_model(torch.cat([h, y], dim=1))
        logstd = self.enc_logstd_model(h)
        return mu, logstd


class Decoder(nn.Module):
    def __init__(self, latent_dim, latent_shape, latent_chns, deconv_filter_cfgs,
                 dec_fc_hidden_dims=(64, 128), ):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.latent_shape = latent_shape
        self.latent_chns = latent_chns
        self.deconv_filter_cfs = deconv_filter_cfgs

        self.dec_fc_out_dim = self.latent_shape[0] * self.latent_shape[1] * self.latent_chns
        self.dec_fc_model = MLP(latent_dim, dec_fc_hidden_dims, self.dec_fc_out_dim)
        self.deconv_model = MultiConvLayers(deconv_filter_cfgs) \
            if len(deconv_filter_cfgs) > 0 else None

    def forward(self, z):
        h = F.leaky_relu(self.dec_fc_model(z))
        if self.deconv_model is not None:
            h = h.view(-1,
                       self.latent_chns,
                       self.latent_shape[0],
                       self.latent_shape[1])
            h = self.deconv_model(h)
        return h


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


if __name__ == "__main__":
    # from torchinfo import summary
    #
    # enc_cfg = [{"in_channels": 1, "out_channels": 32, "kernel_size": 3,
    #             "stride": 1, "dilation": 1, "padding": 1},
    #            {"in_channels": 32, "out_channels": 64, "kernel_size": 3,
    #             "stride": 1, "dilation": 1, "padding": 1},
    #            {"in_channels": 64, "out_channels": 128, "kernel_size": 3,
    #             "stride": 2, "dilation": 1, "padding": 1},
    #            {"in_channels": 128, "out_channels": 128, "kernel_size": 3,
    #             "stride": 2, "dilation": 1, "padding": 1},
    #            {"in_channels": 128, "out_channels": 128, "kernel_size": 3,
    #             "stride": 2, "dilation": 1, "padding": 1}, ]
    # m1 = MultiConvLayers(enc_cfg)
    # summary(m1, input_size=(1, 40, 40), device="cpu")
    #
    # dec_cfg = [{"in_channels": 128, "out_channels": 128, "kernel_size": 3, "stride": 1,
    #             "dilation": 1, "padding": 1, "output_padding": 0, "is_deconv": True},
    #            {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "stride": 2,
    #             "dilation": 1, "padding": 1, "output_padding": 1, "is_deconv": True},
    #            {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "stride": 1,
    #             "dilation": 1, "padding": 1, "output_padding": 0, "is_deconv": True},
    #            {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "stride": 2,
    #             "dilation": 1, "padding": 1, "output_padding": 1, "is_deconv": True},
    #            {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "stride": 1,
    #             "dilation": 1, "padding": 1, "output_padding": 0, "is_deconv": True},
    #            {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "stride": 2,
    #             "dilation": 1, "padding": 1, "output_padding": 1, "is_deconv": True,
    #             "activation": "sigmoid"},
    #            ]
    # m2 = MultiConvLayers(dec_cfg)
    # summary(m2, input_size=(128, 5, 5), device="cpu")
    out = compute_conv2dTranspose_output_shape(
        h_in=64, w_in=64, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2),
        output_padding=0, dilation=1
    )
    print(out)

import antialiased_cnns
import numpy as np
import torch
import torch.nn as nn

from .transformer import Transformer


class SRTConvBlock(nn.Module):
    def __init__(self, idim, hdim=None, odim=None):
        super().__init__()
        if hdim is None:
            hdim = idim

        if odim is None:
            odim = 2 * hdim

        conv_kwargs = {"bias": False, "kernel_size": 3, "padding": 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2, **conv_kwargs),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


GROUP_NORM_LOOKUP = {
    16: 2,  # -> channels per group: 8
    32: 4,  # -> channels per group: 8
    64: 8,  # -> channels per group: 8
    128: 8,  # -> channels per group: 16
    256: 16,  # -> channels per group: 16
    512: 32,  # -> channels per group: 16
    1024: 32,  # -> channels per group: 32
    2048: 32,  # -> channels per group: 64
}


# Recursively traverse through modules and replace Batchnorms with Groupnorms
def replace_bn(layer, stray_bn=False):
    if isinstance(layer, nn.BatchNorm2d) and not stray_bn:
        num_channels = layer.num_features

        return nn.GroupNorm(GROUP_NORM_LOOKUP[num_channels], num_channels)

    for name, module in layer.named_modules():
        if name:
            try:
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    layer._modules[name] = nn.GroupNorm(
                        GROUP_NORM_LOOKUP[num_channels], num_channels
                    )
            except AttributeError:
                name = name.split(".")[0]
                sub_layer = getattr(layer, name)
                sub_layer = replace_bn(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)

    return layer


def get_resnet(
    feature_dim=2048, use_group_norm=True, num_patches=1, relpose=False, stray_bn=False
):
    """
    In: (B, 3, 224, 224) | Out: (B, 2048, 1, 1).
    """

    if relpose:
        model = antialiased_cnns.resnet50(pretrained=True)
        layers = list(model.children())[:9]
        feature_extractor = nn.Sequential(*layers)
        return feature_extractor

    if feature_dim == 512:
        model = antialiased_cnns.resnet18(pretrained=True)
    else:
        model = antialiased_cnns.resnet50(pretrained=True)

    if num_patches <= 7:
        layers = list(model.children())[:8]
        layers.append(nn.AdaptiveAvgPool2d((num_patches, num_patches)))
    else:
        layers = list(model.children())[:7]
        layers.append(nn.Conv2d(1024, 2048, kernel_size=(1, 1), bias=False))
        layers.append(nn.BatchNorm2d(2048))
        layers.append(nn.Conv2d(2048, 2048, kernel_size=(1, 1), bias=False))
        layers.append(nn.BatchNorm2d(2048))
        layers.append(nn.AdaptiveAvgPool2d((num_patches, num_patches)))

    if use_group_norm:
        for i in range(len(layers)):
            layers[i] = replace_bn(layers[i], stray_bn=stray_bn)

    feature_extractor = nn.Sequential(*layers)
    return feature_extractor


def get_srt_patch_version(num_patches, feature_dim=512):
    if num_patches == 5:
        num_blocks = 4
        final_kernel = 3

    layers = [SRTConvBlock(idim=3, hdim=9)]
    cur_hdim = 18
    for i in range(1, num_blocks):
        layers.append(SRTConvBlock(idim=cur_hdim, odim=None))
        cur_hdim *= 2

    last = nn.Conv2d(cur_hdim, feature_dim, kernel_size=final_kernel)
    per_patch_linear = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    layers.append(last)
    layers.append(per_patch_linear)

    return nn.Sequential(*layers)


def get_srt_encoder(num_patches):
    if num_patches == 7:
        num_blocks = 5
        final_kernel = 1
    elif num_patches == 3:
        num_blocks = 6
        final_kernel = 2
    elif num_patches == 5:
        num_blocks = 5
        final_kernel = 3

    layers = [SRTConvBlock(idim=3, hdim=9)]
    cur_hdim = 18
    for i in range(1, num_blocks):
        layers.append(SRTConvBlock(idim=cur_hdim, odim=None))
        cur_hdim *= 2

    last = nn.Conv2d(cur_hdim, 512, kernel_size=final_kernel)
    per_patch_linear = nn.Conv2d(512, 512, kernel_size=1)

    layers.append(last)
    layers.append(per_patch_linear)

    return nn.Sequential(*layers)


class FeaturePositionalEncoding(nn.Module):
    def _get_sinusoid_encoding_table(self, n_position, d_hid, base):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(base, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def __init__(self, num_images=2, feature_dim=2048, num_patches=1):
        super().__init__()
        self.num_images = num_images
        self.feature_dim = feature_dim
        self.num_patches = num_patches
        self.num_sub_tokens = 1

        self.register_buffer(
            "pos_table_1",
            self._get_sinusoid_encoding_table(self.num_images, self.feature_dim, 10000),
        )

        if self.num_patches > 1:
            self.num_sub_tokens = self.num_patches * self.num_patches + 1
            self.register_buffer(
                "pos_table_2",
                self._get_sinusoid_encoding_table(
                    self.num_sub_tokens, self.feature_dim, 70007
                ),
            )

    def forward(self, x):
        batch_size = x.shape[0]
        num_tokens = x.shape[1] // self.num_sub_tokens

        if self.num_patches == 1:
            pe = self.pos_table_1[:, :num_tokens].clone().detach()
            x_pe = x + pe

        else:
            x = x.reshape(batch_size, num_tokens, self.num_sub_tokens, self.feature_dim)

            # To encode image #
            pe1 = self.pos_table_1[:, :num_tokens].clone().detach()
            pe1 = pe1.reshape((1, num_tokens, 1, self.feature_dim))
            pe1 = pe1.repeat((batch_size, 1, self.num_sub_tokens, 1))

            # To encode patch #
            pe2 = self.pos_table_2.clone().detach()
            pe2 = pe2.reshape((1, 1, self.num_sub_tokens, self.feature_dim))
            pe2 = pe2.repeat((batch_size, num_tokens, 1, 1))

            x_pe = x + pe1 + pe2
            x_pe = x_pe.reshape(
                (batch_size, num_tokens * self.num_sub_tokens, self.feature_dim)
            )

        return x_pe


class GlobalFeatures(nn.Module):
    def __init__(self, num_images=2, feature_dim=2048, depth=8, stray_bn=False):
        super().__init__()
        self.num_images = num_images
        self.feature_dim = feature_dim

        self.feature_extractor = get_resnet(
            feature_dim=self.feature_dim, stray_bn=stray_bn
        )
        self.feature_positional_encoding = FeaturePositionalEncoding(
            self.num_images, self.feature_dim, 1
        )
        self.transformer = Transformer(dim=self.feature_dim, depth=depth)
        self.transformer = nn.Sequential(
            self.transformer, nn.LayerNorm(self.feature_dim)
        )

    def forward(self, images, crop_pe=None):
        batch_size = images.shape[0]
        num_tokens = images.shape[1]

        if crop_pe is None:
            features = torch.zeros(
                (batch_size, num_tokens, self.feature_dim), device=images.device
            )
            for i in range(batch_size):
                features[i] = self.feature_extractor(images[i]).reshape(
                    (num_tokens, self.feature_dim)
                )
            features = self.feature_positional_encoding(features)
            features = self.transformer(features)
        else:
            features = torch.zeros(
                (batch_size, num_tokens, self.feature_dim), device=images.device
            )
            for i in range(batch_size):
                resnet_feature = self.feature_extractor(images[i]).squeeze()
                features[i] = torch.cat((resnet_feature, crop_pe[i]), dim=-1)
            features = self.feature_positional_encoding(features)
            features = self.transformer(features)

        return features

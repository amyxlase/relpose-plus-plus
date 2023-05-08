import torch
import torch.nn as nn


class TranslationRegressor(nn.Module):
    def __init__(
        self,
        num_images=2,
        hidden_size_1=1024,
        hidden_size_2=512,
        crop_pe_dim=96,
        feature_dim=2048,
    ):
        """
        Args:
            feature_extractor (nn.Module): Feature extractor.
            num_pe_bases (int): Number of positional encoding bases.
            num_layers (int): Number of layers in the network.
            hidden_size (int): Size of the hidden layer.
            recursion_level (int): Recursion level for healpix if using equivolumetric
                sampling.
            num_queries (int): Number of rotations to sample if using random sampling.
            sample_mode (str): Sampling mode. Can be equivolumetric or random.
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_images = num_images
        self.crop_pe_dim = crop_pe_dim
        self.full_feature_dim = self.feature_dim + self.crop_pe_dim

        self.embed_feature = nn.Linear(2 * self.full_feature_dim, hidden_size_1)

        self.block1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_size_1, hidden_size_1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
        )
        self.block2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_size_2, hidden_size_2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size_2, hidden_size_2),
        )
        self.decode = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_size_2, 3),
        )

        # self.decode[1].weight.data.fill_(1e-5)
        # self.decode[1].weight.data.fill_(1e-5)

    def positional_encoding(self, x):
        """
        Args:
            x (tensor): Input (B, D).

        Returns:
            y (tensor): Positional encoding (B, 2 * D * L).
        """
        embed = (x[..., None] * self.embedding).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

    def forward(
        self,
        features=None,
    ):
        num_batches = features.shape[0]
        num_tokens = features.shape[1]

        # Construct new matrix
        features_full = torch.zeros(
            (num_batches * num_tokens, 2 * self.full_feature_dim)
        ).to(features.device)
        for i in range(num_batches):
            for j in range(num_tokens):
                feature1 = features[i][j]
                feature2 = features[i][0]
                features_full[i * num_tokens + j] = torch.cat(
                    (feature1, feature2), dim=0
                )

        e_f = self.embed_feature(features_full)

        h = self.block1(e_f)
        h = h + self.block2(h)
        translations = self.decode(h).reshape((num_batches * num_tokens, 3))

        bias = torch.FloatTensor([[0, 0, 1]]).to(translations.device)
        translations += bias

        return translations

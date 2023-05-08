import torch
import torch.nn as nn

from utils import generate_random_rotations, get_permutations

from .encoder import GlobalFeatures
from .regressor import TranslationRegressor


def generate_hypotheses(rotations_gt=None, num_queries=50000):
    """
    Args:
        rotations_gt (tensor): Batched rotations (B, N_I, 3, 3).

    Returns:
        hypotheses (tensor): Hypotheses (B, N_I, N_Q, 3, 3).
    """
    batch_size, num_images, _, _ = rotations_gt.shape
    hypotheses = generate_random_rotations(
        (num_queries - 1) * batch_size * num_images, device=rotations_gt.device
    )
    # (B, N_i, N_q - 1, 3, 3)
    hypotheses = hypotheses.reshape(batch_size, num_images, (num_queries - 1), 3, 3)
    # (B, N_i, N_q, 3, 3)
    hypotheses = torch.cat((rotations_gt.unsqueeze(2), hypotheses), dim=2)
    return hypotheses


class RelPose(nn.Module):
    def __init__(
        self,
        num_pe_bases=8,
        num_layers=4,
        hidden_size=256,
        num_queries=50000,
        num_images=2,
        metadata_size=0,
        architecture="global_features",
        num_patches=1,
        patch_first=False,
        feature_dim=2048,
        depth=8,
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
        self.num_queries = num_queries
        self.num_images = num_images
        self.num_permutations = get_permutations(self.num_images)
        self.metadata_size = metadata_size
        self.architecture = architecture
        self.num_patches = num_patches
        self.num_sub_tokens = 1
        self.patch_first = patch_first
        self.feature_dim = feature_dim
        self.depth = depth
        self.num_pe_bases = num_pe_bases
        self.crop_pe_dim = 2 * 3 * self.num_pe_bases  # For one image
        self.full_feature_dim = self.feature_dim + self.crop_pe_dim

        self.feature_extractor = GlobalFeatures(
            self.num_images,
            self.full_feature_dim,
            self.depth,
        )

        self.translation_regressor = TranslationRegressor(
            num_images=self.num_images,
            feature_dim=self.feature_dim,
            crop_pe_dim=self.crop_pe_dim,
        )

        # Configure query positional encoding
        self.use_positional_encoding = num_pe_bases > 0
        if self.use_positional_encoding:
            query_size = num_pe_bases * 2 * 9
            metadata_size = num_pe_bases * 2 * metadata_size
            self.register_buffer(
                "embedding", (2 ** torch.arange(num_pe_bases)).reshape(1, 1, -1)
            )
        else:
            query_size = 9

        self.embed_feature = nn.Linear(2 * self.full_feature_dim, hidden_size)
        self.embed_query = nn.Linear(query_size, hidden_size)
        if self.metadata_size > 0:
            self.embed_metadata = nn.Linear(metadata_size, hidden_size)
        layers = []
        for _ in range(num_layers - 2):
            layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Linear(hidden_size, 1))
        self.layers = nn.Sequential(*layers)

    def positional_encoding(self, x):
        """
        Args:
            x (tensor): Input (B, D).

        Returns:
            y (tensor): Positional encoding (B, 2 * D * L).
        """
        if not self.use_positional_encoding:
            return x
        embed = (x[..., None] * self.embedding).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

    def generate_queries(self, gt_rotation, device):
        queries = generate_random_rotations(self.num_queries, device=device)
        queries = queries.reshape((self.num_queries, 3, 3))
        # Rotate all queries so first entry will be gt_rotation
        if gt_rotation is not None:
            delta_rot = queries[0].T @ gt_rotation
            queries = torch.einsum("aij,bjk->baik", queries, delta_rot)

        return queries

    def forward(
        self,
        gt_rotation=None,
        queries=None,
        images=None,
        crop_params=None,
        take_softmax=True,
        eval_time=False,
    ):
        """
        Args:
            gt_rotation (tensor): Ground truth rotation (B, 3, 3).
            queries (tensor): Rotation matrices (B, num_queries, 3, 3). If None, will
                be generated randomly.
            images (tensor): Corresponding set of images (B, N, 3, 224, 224).
            num_queries (int): Number of rotations to sample if using random sampling.

        Returns:
            rotations (tensor): Rotation matrices (B, num_queries, 3, 3). First query
                is the ground truth rotation.
            logits (tensor): logits (B, num_queries).
        """
        batch_size = images.shape[0]
        num_tokens = images.shape[1]

        crop_pe = self.positional_encoding(crop_params)
        features = self.feature_extractor(images, crop_pe=crop_pe)

        permutations = get_permutations(num_tokens, eval_time=eval_time)
        queries_to_return = self.num_queries
        logits = torch.zeros(
            (batch_size, len(permutations), queries_to_return),
            device=features.device,
        )
        queries_all = torch.zeros(
            (batch_size, queries_to_return, len(permutations), 3, 3),
            device=features.device,
        )

        for k, (i, j) in enumerate(permutations, start=0):
            a, b = i * self.num_sub_tokens, j * self.num_sub_tokens
            features1 = features[:, a, :]
            features2 = features[:, b, :]

            featuresPair = torch.stack([features1, features2], dim=1)
            featuresPair = featuresPair.reshape(
                (batch_size, (2 * self.full_feature_dim))
            )

            # Generate Queries
            gt = None if gt_rotation is None else gt_rotation[:, k, :, :]
            if queries is None:
                queries_curr = self.generate_queries(gt, images.device)
            else:
                queries_curr = queries[:, :, k, :, :]
            queries_reshaped = queries_curr.reshape(batch_size, self.num_queries, 9)
            queries_pe = self.positional_encoding(queries_reshaped)

            e_f = self.embed_feature(featuresPair).unsqueeze(1)
            e_q = self.embed_query(queries_pe)
            out = self.layers(e_f + e_q)
            out = out.reshape((batch_size, self.num_queries))
            if take_softmax:
                out = torch.softmax(out, dim=1)

            logits[:, k, :] = out
            queries_all[:, :, k, :, :] = queries_curr

        translations = self.translation_regressor(features)
        return queries_all, logits, translations

    def predict_probability(
        self, feature1, feature2, queries=None, rot_zero=None, take_softmax=True
    ):
        featuresPair = torch.stack([feature1, feature2], dim=0)
        featuresPair = featuresPair.reshape((1, 2 * self.full_feature_dim))

        # Generate Queries
        gt = None if rot_zero is None else rot_zero
        if queries is None:
            queries = self.generate_queries(gt, feature1.device)
        queries_reshaped = queries.reshape(1, self.num_queries, 9)
        queries_pe = self.positional_encoding(queries_reshaped)

        # Layers
        e_f = self.embed_feature(featuresPair).unsqueeze(1)
        e_q = self.embed_query(queries_pe)

        out = self.layers(e_f + e_q)
        out = out.reshape((1, self.num_queries))

        if take_softmax:
            out = torch.softmax(out, dim=1)

            probabilities = out[0].detach().cpu().numpy()
            best_prob = probabilities.argmax()
            best_rotation = queries.squeeze().detach().cpu().numpy()[best_prob]
            max_prob = probabilities.max().item()

            return queries, out, best_rotation, max_prob

        else:
            return queries, out, None, None

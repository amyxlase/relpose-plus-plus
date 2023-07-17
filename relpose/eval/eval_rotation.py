import json
import os
import os.path as osp

import numpy as np
import torch
from tqdm.auto import tqdm

from models.util import get_model
from utils import generate_random_rotations, get_permutations

from .util import compute_angular_error_batch, get_dataset

# Cached rotations for deterministic evaluation. If path doesn't exist, random rotations
# will be used.
PROPOSALS_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "../rotations.pt")
# Pre-computed random orders for each sequence
ORDER_PATH = "data/co3d_v2_random_order_{sample_num}/{category}.json"
# Number of samples
NUM_PAIRWISE_QUERIES = 500_000


def get_n_features(model, num_frames, images, crop_params):
    crop_pe = model.positional_encoding(crop_params)
    features = model.feature_extractor(images, crop_pe=crop_pe)
    return features.reshape((1, num_frames, model.full_feature_dim, 1, 1))


# Permutations contain exactly half of the adjacency matrix
def initialize_graph(num_frames, model, images, crop_params):
    if osp.exists(PROPOSALS_PATH):
        proposals = torch.load(PROPOSALS_PATH).to("cuda")
    else:
        proposals = generate_random_rotations(NUM_PAIRWISE_QUERIES, "cuda")
    features = get_n_features(model, num_frames, images, crop_params=crop_params)
    best_rotations = np.zeros((num_frames, num_frames, 3, 3))
    best_probs = np.zeros((num_frames, num_frames))

    for i in range(num_frames):
        for j in range(num_frames):
            if i == j:
                continue

            a, b = min(i, j), max(i, j)
            _, _, best_rotation, best_prob = model.predict_probability(
                features[0, a], features[0, b], queries=proposals
            )

            if i > j:
                best_rotation = best_rotation.T

            best_rotations[i][j] = best_rotation
            best_probs[i][j] = best_prob

    return best_rotations, best_probs


def n_to_np_rotations(num_frames, n_rots):
    R_pred_rel = []
    permutations = get_permutations(num_frames, eval_time=True)
    for i, j in permutations:
        R_pred_rel.append(n_rots[i].T @ n_rots[j])
    R_pred_rel = np.stack(R_pred_rel)

    return R_pred_rel


def compute_mst(num_frames, best_probs, best_rotations):
    # Construct MST using Prim's algorithm (modified for directed graph)
    # Currently a naive O(N^3) implementation :p
    current_assigned = {0}
    assigned_rotations = np.tile(np.eye(3), [num_frames, 1, 1])
    edges = []

    while len(current_assigned) < num_frames:
        best_i = -1
        best_j = -1
        best_p = -1
        not_assigned = set(range(num_frames)) - current_assigned

        # Find the highest probability edge that connects an unassigned node to the MST
        for i in current_assigned:
            for j in not_assigned:
                if best_probs[i, j] > best_p:
                    best_p = best_probs[i, j]
                    best_i = i
                    best_j = j
                if best_probs[j, i] > best_p:
                    best_p = best_probs[j, i]
                    best_i = j
                    best_j = i

        # Once edge is found, keep and mark assigned
        rot = best_rotations[best_i, best_j]
        if best_i in current_assigned:
            current_assigned.add(best_j)
            assigned_rotations[best_j] = assigned_rotations[best_i] @ rot
        else:
            current_assigned.add(best_i)
            assigned_rotations[best_i] = assigned_rotations[best_j] @ rot.T

        edges.append((best_i, best_j))

    return assigned_rotations, edges


def score_hypothesis(
    model,
    num_frames,
    images,
    hypothesis,
    queries,
    crop_params=None,
):
    permutations = get_permutations(num_frames, eval_time=True)
    features = get_n_features(model, num_frames, images, crop_params)

    score = 0
    for i, j in permutations:
        queries[0] = torch.tensor(hypothesis[i].T @ hypothesis[j]).to(features.device)

        with torch.no_grad():
            a, b = min(i, j), max(i, j)
            R = queries if i < j else queries.transpose(1, 2)
            _, logits, _, _ = model.predict_probability(
                feature1=features[0, a],
                feature2=features[0, b],
                queries=R,
                take_softmax=False,
            )

            score += logits[0][0].item()
    return score


def coordinate_ascent(
    num_frames,
    model,
    images,
    crop_params,
    initial_hypothesis,
    num_iterations=50,
    num_queries=250_000,
    use_pbar=True,
):
    model.num_queries = num_queries
    device = next(model.parameters()).device
    hypothesis = torch.from_numpy(initial_hypothesis).to(device).float()
    features = get_n_features(model, num_frames, images, crop_params)

    # scoreslist = []

    it = tqdm(range(num_iterations)) if use_pbar else range(num_iterations)
    for _ in it:
        # Randomly sample an index to update
        k = np.random.choice(num_frames)
        proposals = generate_random_rotations(num_queries, device)
        proposals[0] = hypothesis[k]
        scores = torch.zeros(1, num_queries, device=device)
        for i in range(num_frames):
            if i == k:
                continue

            with torch.no_grad():
                a, b = min(i, k), max(i, k)

                R_rel = hypothesis[i].T @ proposals
                R = R_rel if i < k else R_rel.transpose(1, 2)

                _, logits, _, _ = model.predict_probability(
                    feature1=features[0, a],
                    feature2=features[0, b],
                    queries=R,
                    take_softmax=False,
                )

                scores += logits
                scores += logits

        best_ind = scores.argmax()
        hypothesis[k] = proposals[best_ind]

    return hypothesis


def evaluate_pairwise(
    model,
    images,
    crop_params,
):
    best_probs = []
    rotations_pred = []

    num_frames = images.shape[1]
    permutations = get_permutations(num_frames, eval_time=True)
    features = get_n_features(model, num_frames, images, crop_params)

    if osp.exists(PROPOSALS_PATH):
        proposals = torch.load(PROPOSALS_PATH).to(images.device)
    else:
        proposals = generate_random_rotations(NUM_PAIRWISE_QUERIES, images.device)

    for i, j in permutations:
        model.num_queries = NUM_PAIRWISE_QUERIES

        a, b = min(i, j), max(i, j)
        _, _, best_rotation, best_prob = model.predict_probability(
            features[0, a], features[0, b], queries=proposals
        )

        if i > j:
            best_rotation = best_rotation.T

        rotations_pred.append(best_rotation)
        best_probs.append(best_prob)

    R_pred_rel = np.stack(rotations_pred)
    return R_pred_rel, []


def evaluate_mst(
    model,
    images,
    crop_params=None,
):
    num_frames = images.shape[1]

    model.num_queries = 500_000

    best_rots, best_probs = initialize_graph(
        num_frames,
        model,
        images,
        crop_params=crop_params,
    )
    rotations_pred, edges = compute_mst(
        num_frames=num_frames,
        best_probs=best_probs,
        best_rotations=best_rots,
    )

    R_pred_rel = n_to_np_rotations(num_frames, rotations_pred)

    return R_pred_rel, rotations_pred


def evaluate_coordinate_ascent(
    model,
    images,
    crop_params,
    use_pbar=False,
):
    num_frames = images.shape[1]
    model.num_queries = NUM_PAIRWISE_QUERIES

    best_rots, best_probs = initialize_graph(
        num_frames,
        model,
        images,
        crop_params,
    )

    rotations_pred, _ = compute_mst(
        num_frames=num_frames,
        best_probs=best_probs,
        best_rotations=best_rots,
    )

    hypothesis = coordinate_ascent(
        num_frames,
        model,
        images,
        crop_params,
        rotations_pred,
        use_pbar=use_pbar,
    )
    hypothesis = hypothesis.detach().cpu().numpy()

    R_pred_rel = n_to_np_rotations(num_frames, hypothesis)
    return R_pred_rel, hypothesis.tolist()


def evaluate_category_rotation(
    checkpoint_path,
    category,
    mode,
    num_frames,
    use_pbar=False,
    force=False,
    sample_num=0,
    **kwargs,
):
    # Default save directory
    folder_path = f"eval/{mode}-{num_frames:03d}-sample{sample_num}"
    os.makedirs(os.path.join(checkpoint_path, folder_path), exist_ok=True)

    # Check not already existing results
    path = osp.join(checkpoint_path, f"{folder_path}/{category}.json")
    if osp.exists(path) and not force:
        print(f"{path} already exists, skipping")
        with open(path, "r") as f:
            data = json.load(f)
        angular_errors = []
        for d in data.values():
            angular_errors.extend(d["angular_errors"])
        return np.array(angular_errors)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = get_model(model_dir=checkpoint_path, device=device)

    # Load dataset
    dataset = get_dataset(
        category=category,
        num_images=num_frames,
        eval_time=True,
        normalize_cameras=True,
    )

    iterable = tqdm(dataset) if use_pbar else dataset
    all_errors = {}
    angular_errors = []
    f = open(ORDER_PATH.format(sample_num=sample_num, category=category))
    order = json.load(f)
    for metadata in iterable:
        # Load instance data
        sequence_name = metadata["model_id"]
        key_frames = order[sequence_name][:num_frames]
        batch = dataset.get_data(sequence_name=sequence_name, ids=key_frames)

        # Load ground truth rotations
        rotations = batch["relative_rotation"].to(device).unsqueeze(0)
        n_p = len(get_permutations(num_frames, eval_time=True))
        R_gt_rel = rotations.detach().cpu().numpy().reshape((n_p, 3, 3))

        # Inputs
        images = batch["image"].to(device).unsqueeze(0)
        crop_params = batch["crop_params"].to(device).unsqueeze(0)

        # Model forward pass
        EVAL_FN_MAP = {
            "pairwise": evaluate_pairwise,
            "coordinate_ascent": evaluate_coordinate_ascent,
        }
        R_pred_rel, _ = EVAL_FN_MAP[mode](
            model,
            images,
            crop_params,
        )

        # Compute errors
        errors = compute_angular_error_batch(R_pred_rel, R_gt_rel)

        # Append information to be saved
        angular_errors.extend(errors)
        all_errors[sequence_name] = {
            "R_pred_rel": R_pred_rel.tolist(),
            "R_gt_rel": R_gt_rel.tolist(),
            "angular_errors": errors.tolist(),
            "key_frames": key_frames,  # .tolist(),
        }

    # Save to file
    with open(path, "w") as f:
        json.dump(all_errors, f)

    print(np.mean(np.array(angular_errors) < 15))
    print(np.mean(np.array(angular_errors) < 30))
    return np.array(angular_errors)

import numpy as np
import torch
from tqdm.auto import tqdm

from utils import generate_random_rotations, get_permutations


def get_n_features(model, num_frames, images, crop_params=None):
    crop_pe = model.positional_encoding(crop_params)
    features = model.feature_extractor(images, crop_pe=crop_pe)
    return features.reshape((1, num_frames, model.full_feature_dim, 1, 1))


def initialize_graph(num_frames, model, images, crop_params, use_all_features=False):
    proposals = generate_random_rotations(500_000, images.device)
    features = get_n_features(model, num_frames, images, crop_params=crop_params)
    best_rotations = np.zeros((num_frames, num_frames, 3, 3))
    best_probs = np.zeros((num_frames, num_frames))

    for i in range(num_frames):
        for j in range(num_frames):
            if i == j:
                continue
            if not use_all_features:
                pair = torch.stack((images[0, i], images[0, j]), dim=0).unsqueeze(0)
                crop_param = torch.stack(
                    (crop_params[0, i], crop_params[0, j]), dim=0
                ).unsqueeze(0)
                features = get_n_features(model, 2, pair, crop_param)
                _, _, best_rotation, best_prob = model.predict_probability(
                    feature1=features[0, 0],
                    feature2=features[0, 1],
                    queries=proposals,
                )
            else:
                if i < j:
                    _, _, best_rotation, best_prob = model.predict_probability(
                        feature1=features[0, i],
                        feature2=features[0, j],
                        queries=proposals,
                    )
                else:
                    _, _, best_rotation, best_prob = model.predict_probability(
                        feature1=features[0, j],
                        feature2=features[0, i],
                        queries=proposals,
                    )
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


def correct_joint(
    num_frames,
    model,
    images,
    initial_hypothesis,
    num_iterations=50,
    num_queries=250_000,
    use_pbar=True,
    use_all_features=False,
    crop_params=None,
):
    model.num_queries = num_queries
    device = next(model.parameters()).device
    hypothesis = torch.from_numpy(initial_hypothesis).to(device).float()
    features = get_n_features(model, num_frames, images, crop_params)

    it = tqdm(range(num_iterations)) if use_pbar else range(num_iterations)
    for j in it:
        # Randomly sample an index to update
        k = np.random.choice(num_frames)
        proposals = generate_random_rotations(num_queries, device)
        proposals[0] = hypothesis[k]
        scores = torch.zeros(1, num_queries, device=device)
        for i in range(num_frames):
            if i == k:
                continue

            R_rel = hypothesis[i].T @ proposals
            with torch.no_grad():
                if not use_all_features:
                    pair = torch.stack((images[0, i], images[0, k]), dim=0).unsqueeze(0)
                    crop_param = torch.stack(
                        (crop_params[0, i], crop_params[0, k]), dim=0
                    ).unsqueeze(0)
                    features = get_n_features(model, 2, pair, crop_param)
                    _, logits, _, _ = model.predict_probability(
                        feature1=features[0, 0],
                        feature2=features[0, 1],
                        queries=R_rel,
                        take_softmax=False,
                    )
                    scores += logits

                    pair = torch.stack((images[0, k], images[0, i]), dim=0).unsqueeze(0)
                    crop_param = torch.stack(
                        (crop_params[0, k], crop_params[0, i]), dim=0
                    ).unsqueeze(0)
                    features = get_n_features(model, 2, pair, crop_param)
                    _, logits, _, _ = model.predict_probability(
                        feature1=features[0, 0],
                        feature2=features[0, 1],
                        queries=R_rel.transpose(1, 2),
                        take_softmax=False,
                    )
                    scores += logits
                else:
                    if i < k:
                        _, logits, _, _ = model.predict_probability(
                            feature1=features[0, i],
                            feature2=features[0, k],
                            queries=R_rel,
                            take_softmax=False,
                        )
                    else:
                        _, logits, _, _ = model.predict_probability(
                            feature1=features[0, k],
                            feature2=features[0, i],
                            queries=R_rel.transpose(1, 2),
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
    use_all_features=False,
    crop_params=None,
    full_test_2=False,
    edges=None,
):
    best_probs = []
    rotations_pred = []

    num_frames = images.shape[1]
    permutations = get_permutations(num_frames, eval_time=True)
    features = get_n_features(model, num_frames, images, crop_params)

    proposals = generate_random_rotations(500_000, images.device)

    for i, j in permutations:
        model.num_queries = 500_000

        if not use_all_features:
            if full_test_2 and i > j:
                pair = torch.stack((images[0, j], images[0, i]), dim=0).unsqueeze(0)
                crop_param = torch.stack(
                    (crop_params[0, j], crop_params[0, i]), dim=0
                ).unsqueeze(0)

            else:
                pair = torch.stack((images[0, i], images[0, j]), dim=0).unsqueeze(0)
                crop_param = torch.stack(
                    (crop_params[0, i], crop_params[0, j]), dim=0
                ).unsqueeze(0)

            features = get_n_features(model, 2, pair, crop_param)
            _, _, best_rotation, best_prob = model.predict_probability(
                features[0, 0], features[0, 1], queries=proposals
            )

            if full_test_2 and i > j:
                best_rotation = best_rotation.T

        else:
            if i > j:
                _, _, best_rotation, best_prob = model.predict_probability(
                    features[0, j], features[0, i], queries=proposals
                )
                best_rotation = best_rotation.T

            else:
                _, _, best_rotation, best_prob = model.predict_probability(
                    features[0, i], features[0, j], queries=proposals
                )

        rotations_pred.append(best_rotation)
        best_probs.append(best_prob)

    R_pred_rel = np.stack(rotations_pred)
    return R_pred_rel, []


def evaluate_mst(
    model,
    images,
    use_all_features=False,
    crop_params=None,
    full_test_2=False,
    edges=None,
):
    num_frames = images.shape[1]

    model.num_queries = 500_000

    best_rots, best_probs = initialize_graph(
        num_frames,
        model,
        images,
        crop_params=crop_params,
        use_all_features=use_all_features,
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
    use_all_features=False,
    crop_params=None,
    full_test_2=False,
    edges=None,
):
    num_frames = images.shape[1]
    model.num_queries = 500_000

    best_rots, best_probs = initialize_graph(
        num_frames,
        model,
        images,
        crop_params=crop_params,
        use_all_features=use_all_features,
    )
    rotations_pred, edges = compute_mst(
        num_frames=num_frames,
        best_probs=best_probs,
        best_rotations=best_rots,
    )

    hypothesis = correct_joint(
        num_frames,
        model,
        images,
        rotations_pred,
        use_all_features=use_all_features,
        crop_params=crop_params,
    )
    hypothesis = hypothesis.detach().cpu().numpy()

    R_pred_rel = n_to_np_rotations(num_frames, hypothesis)
    return R_pred_rel, hypothesis.tolist()

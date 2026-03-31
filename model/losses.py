import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.3

    def forward(self, anchor, positive, negative):

        pos_dist = torch.sqrt((anchor - positive).pow(2).sum())
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))

        loss = F.relu(pos_dist-neg_dist + self.margin)
        return loss#.mean()

class FocusInfoNCELoss(nn.Module):
    def __init__(self, tau=0.2):
        super().__init__()
        self.tau = tau  # Temperature parameter
        self.pos_num = 192

    def forward(self, features_a, features_b, matches, focus_points, matched_index):
        """
        Args:
            features_a: Patch features of A, shape [H, W, C]
            features_b: Patch features of B, shape [H, W, C]
            matches: Match matrix [M, 4], each row [Ax, Ay, Bx, By]
            focus_points: Patch coordinates in B to focus on [N_b, 2], each row [Bx, By]
            matched_index: Index of each match in focus_points [M]
        """
        H, W, C = features_a.shape  # Feature map size of A
        assert features_b.shape == (H, W, C), "Feature map sizes of A and B must be the same"

        # print(matches.shape, focus_points.shape, matched_index.shape)
        features_a = features_a.view(-1, C)  # [H*W, C]
        features_b = features_b.view(-1, C)  # [H*W, C]

        if matches.shape[0] > self.pos_num:
            # If matches exceed pos_num, randomly sample
            indices = torch.randperm(matches.shape[0])[:self.pos_num]
            matches = matches[indices]
            matched_index = matched_index[indices]

        # 1. Extract patch features of all positive samples in A
        a_indices = matches[:, 0] * W + matches[:, 1]  # Patch index of A [M]
        pos_features_a = features_a[a_indices]  # [M, C]


        # 2. Extract patch features of all focus_points in B
        # Convert focus_points coordinates to 1D indices
        focus_indices = focus_points[:, 0] * W + focus_points[:, 1]  # [N_b]
        focus_features_b = features_b[focus_indices]  # [N_b, C]

        # 3. Normalize features (cosine similarity requires unit vectors)
        pos_features_a = F.normalize(pos_features_a, dim=1)
        focus_features_b = F.normalize(focus_features_b, dim=1)

        # 4. Build similarity matrix (M x N_b)
        similarities = torch.matmul(pos_features_a, focus_features_b.T) / self.tau  # [M, N_b]

        # 5. Define labels (positive sample for each A patch is matched_index[i])
        labels = matched_index  # [M], each element is the position index of positive sample in focus_points

        # 6. Compute cross-entropy loss (InfoNCE)
        loss = F.cross_entropy(similarities, labels)
        return loss


class HardInfoNCELoss(nn.Module):
    def __init__(self, tau=0.2, top_k=100):
        super().__init__()
        self.tau = tau  # Temperature parameter
        self.top_k = top_k  # Number of top-K hard negative samples to keep for each positive sample
        self.pos_num = 192

    def forward(self, features_a, features_b, matches, focus_points, matched_index):
        """
        Args:
            features_a: Features of image A, shape [H, W, C]
            features_b: Features of image B, shape [H, W, C]
            matches: Match matrix [M, 4], each row [Ax, Ay, Bx, By]
            focus_points: Patch coordinates in B to focus on [N_b, 2]
            matched_index: Index of each match in focus_points [M]
        """
        H, W, C = features_a.shape  # Feature map size of A
        assert features_b.shape == (H, W, C), "Feature map sizes of A and B must be the same"

        features_a = features_a.view(-1, C)  # [H*W, C]
        features_b = features_b.view(-1, C)  # [H*W, C]
        # print(matches.shape, focus_points.shape, matched_index.shape)

        if matches.shape[0] > self.pos_num:
            # If matches exceed pos_num, randomly sample
            indices = torch.randperm(matches.shape[0])[:self.pos_num]
            matches = matches[indices]
            matched_index = matched_index[indices]

        # 1. Extract patch features of all positive samples in A
        a_indices = matches[:, 0] * W + matches[:, 1]  # Patch index of A [M]
        pos_features_a = features_a[a_indices]  # [M, C]

        # 2. Extract patch features of all focus_points in B
        focus_indices = focus_points[:, 0] * W + focus_points[:, 1]  # [N_b]
        focus_features_b = features_b[focus_indices]  # [N_b, C]

        # 3. Normalize features (cosine similarity requires unit vectors)
        pos_features_a = F.normalize(pos_features_a, dim=1)
        focus_features_b = F.normalize(focus_features_b, dim=1)

        # 4. Build similarity matrix (M x N_b)
        similarities = torch.matmul(pos_features_a, focus_features_b.T) / self.tau  # [M, N_b]
        # print("similarities:", similarities)
        # 5. Exclude positive sample positions (negative samples for each positive are all focus_points except matched_index)
        # Create mask to exclude positive sample positions
        mask = torch.ones_like(similarities, dtype=torch.bool)
        mask[torch.arange(len(matched_index)), matched_index] = False  # Exclude positive sample positions

        # 6. Mine hard negative samples (keep top-K negative samples with highest similarity)
        # Extract similarities of all negative samples
        neg_similarities = similarities[mask].view(len(similarities), -1)  # [M, N_b-1]
        # Select top-K hard examples (K negative samples with highest similarity)
        real_topk = min(self.top_k, neg_similarities.shape[1]) # Ensure top-K doesn't exceed actual negative sample count
        _, top_k_indices = torch.topk(neg_similarities, k=real_topk, dim=1, largest=True)  # [M, K]

        # print("top_k_indices:", top_k_indices)

        # 7. Build new similarity matrix (only containing positive samples and top-K hard negative samples)
        # Extract features of top-K negative samples
        hard_neg_features_b = focus_features_b[top_k_indices]  # [M, K, C]

        # Concatenate positive samples and hard negative samples
        pos_and_hard_neg_features_b = torch.cat([
            pos_features_a.unsqueeze(1),  # [M, 1, C]
            hard_neg_features_b  # [M, K, C]
        ], dim=1)  # [M, K+1, C]

        # Compute new similarity matrix (M x (K+1))
        new_similarities = torch.bmm(
            pos_features_a.unsqueeze(1),  # [M, 1, C]
            pos_and_hard_neg_features_b.transpose(1, 2)  # [M, C, K+1]
        ).squeeze(1) / self.tau  # [M, K+1]

        # 8. Define labels (positive sample is always at the first position)
        labels = torch.zeros(len(new_similarities), dtype=torch.long).to(new_similarities.device)  # [M]

        # 9. Compute cross-entropy loss (only for positive samples and hard negative samples)
        loss = F.cross_entropy(new_similarities, labels)
        return loss

class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features_a, features_b, matches):
        H, W, C = features_a.shape  # Feature map size of A
        assert features_b.shape == (H, W, C), "Feature map sizes of A and B must be the same"
        features_a = features_a.view(-1, C)  # [H*W, C]
        features_b = features_b.view(-1, C)

        pos_indices_a = matches[:, 0] * W + matches[:, 1]
        pos_indices_b = matches[:, 2] * W + matches[:, 3]
        pos_a = features_a[pos_indices_a]  # [M, C]
        pos_b = features_b[pos_indices_b]  # [M, C]

        # Compute cosine similarity loss directly
        cos_sim = F.cosine_similarity(pos_a, pos_b, dim=1)
        loss = 1 - cos_sim.mean()  # Maximize similarity
        return loss

if __name__ == "__main__":
    # Test HardInfoNCELoss
    hard_loss = HardInfoNCELoss(tau=0.2, top_k=50)
    cos_loss = CosineLoss()
    features_a = torch.randn(40, 40, 3)  # Feature map of A [H, W, C]
    features_b = torch.randn(40, 40, 3)  # Feature map of B [H, W, C]

    # features_a = torch.ones(16, 16, 3)  # Feature map of A [H, W, C]
    # features_b = torch.ones(16, 16, 3)  # Feature map of B [H, W, C]

    # features_a[2, 3, :] = torch.tensor([10, 0, 0])  # Feature at position (2,3) of A
    # features_a[4, 1, :] = torch.tensor([0, 5, 0])  # Feature at position (4,1) of A
    # features_a[0, 0, :] = torch.tensor([0, 0, 5])  # Feature at position (0,0) of A
    # features_b[5, 6, :] = torch.tensor([5, 0, 0])  # Feature at position (5,6) of B
    # features_b[7, 2, :] = torch.tensor([0, 5, 0])  # Feature at position (7,2) of B
    # features_b[13, 13, :] = torch.tensor([0, 0, 5])  # Feature at position (13,13) of B


    focus_points = torch.tensor([
        [5, 6],  # Example coordinates
        [7, 2],
        [13, 13],
        [1, 1],  # Non-matching focus point
        [3, 4]
    ])

    # Match matrix (example: 3 match pairs)
    matches = torch.tensor([
        [2, 3, 5, 6],  # A's (2,3) matches B's (5,6)
        [4, 1, 7, 2],
        [0, 0, 13, 13]
    ])

    # matched_index (index of each match in focus_points)
    matched_index = torch.tensor([0, 1, 2])  # e.g., first match corresponds to focus_points[0]

    # loss = hard_loss(features_a, features_b, matches, focus_points, matched_index)
    loss = cos_loss(features_a, features_b, matches)
    print("Loss:", loss.item())

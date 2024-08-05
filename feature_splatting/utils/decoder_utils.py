import torch
import torch.nn as nn
import torch.nn.functional as F

class two_layer_mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim_dict):
        super(two_layer_mlp, self).__init__()
        self.hidden_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        feature_branch_dict = {}
        for key, feat_dim_chw in feature_dim_dict.items():
            feature_branch_dict[key] = nn.Conv2d(hidden_dim, feat_dim_chw[0], kernel_size=1, stride=1, padding=0)
        self.feature_branch_dict = nn.ModuleDict(feature_branch_dict)

    def forward(self, x):
        intermediate_feature = self.hidden_conv(x)
        intermediate_feature = F.relu(intermediate_feature)
        ret_dict = {}
        for key, nn_mod in self.feature_branch_dict.items():
            ret_dict[key] = nn_mod(intermediate_feature)
        return ret_dict
    
    @torch.no_grad()
    def per_gaussian_forward(self, x):
        intermediate_feature = F.linear(x, self.hidden_conv.weight.view(self.hidden_conv.weight.size(0), -1), self.hidden_conv.bias)
        intermediate_feature = F.relu(intermediate_feature)
        ret_dict = {}
        for key, nn_mod in self.feature_branch_dict.items():
            ret_dict[key] = F.linear(intermediate_feature, nn_mod.weight.view(nn_mod.weight.size(0), -1), nn_mod.bias)
        return ret_dict

def compute_similarity(prob_mn, softmax_temp, num_pos, heatmap_method="standard_softmax"):
    """
    Compute probability of an element being positive

    Args:
        prob_mn: Tensor of shape (m, n); where m is the number of total classes; n is the number of elements
        softmax_temp: float
        num_pos: int
    """
    assert num_pos <= prob_mn.shape[0]
    if heatmap_method == "standard_softmax":  # Feature splatting uses this
        prob_mn = prob_mn / softmax_temp
        probs = prob_mn.softmax(dim=0)
        pos_sim = probs[:num_pos].sum(dim=0)  # H, W
        return pos_sim
    elif heatmap_method == "pairwise_softmax":  # F3RM uses this
        # Broadcast positive label similarities to all negative labels
        pos_sims = prob_mn[:num_pos]
        neg_sims = prob_mn[num_pos:]
        pos_sims = pos_sims.mean(dim=0, keepdim=True)
        pos_sims = pos_sims.broadcast_to(neg_sims.shape)
        paired_sims = torch.cat([pos_sims, neg_sims], dim=0)

        # Compute paired softmax
        probs = (paired_sims / softmax_temp).softmax(dim=0)[:1, ...]
        torch.nan_to_num_(probs, nan=0.0)
        sims, _ = probs.min(dim=0)
        return sims
    else:
        raise ValueError(f"Unknown heatmap method: {heatmap_method}")

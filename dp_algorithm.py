import torch
from torch.distributions.gamma import Gamma

def sample_noise_Chi(x, d_shape, eta):
    n_dim = d_shape[-1]
    alpha = torch.ones(d_shape) * n_dim
    beta = torch.ones(d_shape) * eta
    m = Gamma(alpha, beta)
    l_lst = m.sample()
    v_lst = -2 * torch.rand(d_shape) + 1
    noise = l_lst * v_lst
    noise = noise.to(x.device)
    disturbed_x = x + noise
    return disturbed_x, noise

def replace_with_nearest(embeddings, all_embeddings):
    batch_size, max_length, num_feat = embeddings.shape
    distances = torch.cdist(embeddings.view(-1, num_feat), all_embeddings, p=2)
    _, nearest_indices = torch.min(distances, dim=-1)
    nearest_embeddings = all_embeddings[nearest_indices]
    return nearest_embeddings.view(batch_size, max_length, num_feat), nearest_indices.view(batch_size, max_length)

def get_dp_output(client_outputs, input_ids, weight, top_k_tokens_per_class, args):
    dp_client_outputs, noise = sample_noise_Chi(client_outputs, client_outputs.shape, args.dp_eta)
    if args.use_cti == "t":
        ui_values_tensor = torch.tensor([top_k_tokens_per_class[label.item()] for label in y], device=client_outputs.device)
        input_ids_expanded = input_ids.unsqueeze(-1)
        ui_mask = (input_ids_expanded == ui_values_tensor.unsqueeze(1)).any(-1)
        temp_x = torch.where(ui_mask.unsqueeze(-1), client_outputs, noise + client_outputs)
        dp_client_outputs = torch.tensor(temp_x, dtype=torch.float32)
    dp_client_outputs, dp_indices = replace_with_nearest(dp_client_outputs, weight)
    return dp_client_outputs, dp_indices

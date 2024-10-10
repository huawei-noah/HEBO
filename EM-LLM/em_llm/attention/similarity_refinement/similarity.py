import torch

def modularity(A, communities):
    '''
    Inputs:
        - A: Similarity matrix (defined above). 
            Expected dimensions: (... , position, position)
        - communities: List of lists of position indices each defining a single community.
    Outputs: 
        - Matrix of modularity values by layer and head. Dimensions: (layer, head)
    '''
    ndims = len(A.shape)
    m = A.sum(dim=(-2,-1)) / 2

    # Precompute the entire matrix for expected edges
    k_i = A.sum(dim=-2)  # Sum rows for each node i
    k_j = A.sum(dim=-1)  # Sum columns for each node j
    expected_edges = torch.einsum('...i,...j->...ij', k_i, k_j)
    expected_edges /= (2 * m.unsqueeze(-1).unsqueeze(-1)) if ndims > 2 else (2 * m)  # unsqueeze 2 dimensions for broadcasting if not real number

    Q = torch.zeros(A.shape[:-2]) if ndims > 2 else 0.0
    for community in communities:
        sub_A = A[..., community, :][..., :, community]
        sub_expected_edges = expected_edges[..., community, :][..., :, community]
        
        Q += (sub_A - sub_expected_edges).sum(dim=(-2, -1)).cpu()

    return Q / (4 * m.cpu())

def conductance(A, communities):
    conductance = []
    total_vol = torch.sum(A, dim=(-2,-1))
    for community in communities:
        community_bar = [i for i in range(A.shape[-1]) if i not in community]
        cut_edges = torch.sum(A[..., community, :][..., :, community_bar], dim=(-2,-1))

        vol_S = torch.sum(A[..., community, :][..., :, community], dim=(-2,-1))
        vol_S_bar = total_vol - vol_S

        # Avoid division by zero
        min_vol = torch.minimum(vol_S, vol_S_bar)
        min_vol[min_vol == 0] = 1e-15

        conductance.append(cut_edges/min_vol)
        del cut_edges, vol_S, vol_S_bar

    conductance = torch.stack(conductance)
    if len(A.shape) > 2:
        min_conductance = conductance[torch.argmin(conductance.sum(dim=(-2,-1)))]
        max_conductance = conductance[torch.argmax(conductance.sum(dim=(-2,-1)))]
        mean_conductance = conductance.mean(dim=0)
    else:
        min_conductance = conductance.min()
        max_conductance = conductance.max()
        mean_conductance = conductance.mean()

    return min_conductance, max_conductance, mean_conductance, conductance
    
def intra_inter_sim(A, communities, return_mean=True):
    inter = []
    intra = []
    for community in communities:
        intra.append(torch.mean(A[..., community, :][..., :, community], dim=(-2,-1)))
        community_bar = [i for i in range(A.shape[-1]) if i not in community]
        inter.append(torch.mean(A[..., community, :][..., :, community_bar], dim=(-2,-1)))

    ratio = [i/j for i,j in zip(intra, inter)]

    if return_mean:
        intra = torch.mean(torch.stack(intra), dim=0)
        inter = torch.mean(torch.stack(inter), dim=0)
        ratio = torch.mean(torch.stack(ratio), dim=0)

    return ratio, intra, inter

def calc_adjacent_similarity_with_offset(A, first_indx, last_indx, sim_func=modularity):
    '''
    Inputs:
        A: Adjacency matrix
        first_indx/last_indx: The first and last indices to test.
        sim_func: The similarity function to use.
    Outputs:
        torch.Tensor: The similarity values for the range of indices.
    '''
    if first_indx > last_indx or first_indx > A.shape[0] or last_indx > A.shape[0]:
        raise Exception(f'Problem with indices in similarity calculation: {first_indx}, {last_indx}, {A.shape}')
    T = A.shape[0]
    result = torch.zeros(last_indx-first_indx)
    for t in range(first_indx, last_indx):
        communities = [list(range(0,t)),list(range(t,T))]
        result[t-first_indx] = sim_func(A, communities)
    return result

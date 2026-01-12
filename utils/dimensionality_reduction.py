import torch

def four_band_reduction(D: torch.Tensor,
                        num_agents: torch.Tensor) -> torch.Tensor:
    """
    Four-band reduction of a batch of pairwise distance matrices.

    Assumption:
        num_agents[b] > 2 for all batch elements.

    Parameters
    ----------
    D : torch.Tensor
        Shape (B, M, M)
        Batched pairwise distance matrices.

    num_agents : torch.Tensor
        Shape (B,) or (B,1)
        Number of valid agents in each batch element.

    Returns
    -------
    out : torch.Tensor
        Shape (B, 4*M - 10)

        For batch element b:
            The first (4*num_agents[b] - 10) entries contain valid
            four-band distances packed contiguously.
            The remaining entries are zero padding.

        Packing order (row-wise):
            Row i contributes:
                D[b, i, i+1], D[b, i, i+2], D[b, i, i+3], D[b, i, i+4]
            for all valid column indices < num_agents[b].
    """

    B, M, _ = D.shape
    device = D.device

    # Ensure num_agents is shape (B,)
    num_agents = num_agents.view(B).to(device)

    # Maximum possible number of four-band values when n = M
    # (M-1) + (M-2) + (M-3) + (M-4) = 4*M - 10
    max_len = 4 * M - 10

    # Output tensor initialized to zero padding
    out = torch.zeros(B, max_len, device=device)

    # Write pointer for each batch element indicating
    # where the next values should be written
    write_ptr = torch.zeros(B, dtype=torch.long, device=device)

    # Iterate over rows of the distance matrix
    for i in range(M):

        # Candidate forward columns: i+1, i+2, i+3, i+4
        cols = torch.arange(i + 1, i + 5, device=device)     # (4,)

        # For each batch element, check which columns are valid
        # Valid iff column index < num_agents[b]
        valid = cols.view(1, 4) < num_agents.view(B, 1)       # (B,4)

        # Clamp columns for safe indexing (invalid entries masked later)
        cols = cols.clamp(max=M - 1)

        # Gather the 4 candidate distances for every batch
        vals = D[:, i, cols]                                # (B,4)

        # Mask invalid entries
        vals = vals * valid

        # Number of valid values contributed by this row for each batch
        k = valid.sum(dim=1)                                # (B,)

        # Pack contiguously into output tensor
        # (small batch loop is unavoidable for ragged packing)
        for b in range(B):
            kb = int(k[b].item())
            if kb > 0:
                start = write_ptr[b]
                out[b, start : start + kb] = vals[b, :kb]
                write_ptr[b] += kb

    return out


def k_local_reduction_4k(
    D: torch.Tensor,
    num_agents: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    k-local neighborhood reduction with minimal (non-redundant) dimension = 4*k.

    Parameters
    ----------
    D : torch.Tensor
        Shape (B, M, M)
        Batched pairwise distance matrices.

    num_agents : torch.Tensor
        Shape (B,) or (B,1)

    k : int
        Local window size (odd recommended).

    Returns
    -------
    out : torch.Tensor
        Shape (B, M, 4*k)

        For each batch b and agent i:
            out[b, i] is a vector of length 4*k containing
            band-limited distances from the k-window centered at i,
            packed contiguously and zero padded.
    """

    B, M, _ = D.shape
    device = D.device
    num_agents = num_agents.view(B).to(device)

    r = k // 2
    feat_dim = 4 * k

    # Output tensor
    out = torch.zeros(B, M, feat_dim, device=device)

    # Iterate over agents
    for i in range(M):

        # Local window indices centered at i
        window = torch.arange(i - r, i + r + 1, device=device)   # (k,)
        window_len = window.numel()

        # Clip for safe indexing
        window = window.clamp(min=0, max=M - 1)

        # Write pointer per batch for packing
        write_ptr = torch.zeros(B, dtype=torch.long, device=device)

        # For each local row p in the window
        for t in range(window_len):
            p = window[t]

            # Neighbor candidates: second and first neighbors
            nbrs = torch.tensor(
                [p - 2, p - 1, p + 1, p + 2],
                device=device
            )                                                   # (4,)

            # Valid neighbor mask per batch
            valid = (
                (nbrs.view(1, 4) >= 0) &
                (nbrs.view(1, 4) < num_agents.view(B, 1))
            )                                                   # (B,4)

            # Clamp for safe gather
            nbrs = nbrs.clamp(min=0, max=M - 1)

            # Gather distances
            vals = D[:, p, nbrs]                                # (B,4)
            vals = vals * valid                                # mask invalid

            # Number of valid values contributed by this row
            k_local = valid.sum(dim=1)                          # (B,)

            # Pack contiguously
            for b in range(B):
                kb = int(k_local[b].item())
                if kb > 0:
                    start = write_ptr[b]
                    out[b, i, start : start + kb] = vals[b, :kb]
                    write_ptr[b] += kb

        # Remaining entries stay zero padded

    return out

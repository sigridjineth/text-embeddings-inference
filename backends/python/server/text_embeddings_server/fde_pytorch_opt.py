import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class FDEConfig:
    ksim: int
    d_proj: int
    R_reps: int
    d_final: Optional[int] = None
    fill_empty_clusters: bool = True
    seed: int = 42
    use_mixed_precision: bool = False

@dataclass
class SamplerConfig:
    token_sample_ratio: float = 1.0
    per_bucket_cap: int = 0
    use_pps: bool = False
    r_sub: int = 0
    seed: int = 42

def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)

def chamfer_similarity(q: torch.Tensor, d: torch.Tensor) -> float:
    # q: [M, D], d: [N, D]
    # MaxSim: sum_i max_j (q_i . d_j)
    sim_matrix = torch.mm(q, d.t())  # [M, N]
    max_sims, _ = torch.max(sim_matrix, dim=1)  # [M]
    return max_sims.sum().item()

class FDEBuilder:
    def __init__(self, d: int, config: FDEConfig, device: torch.device):
        self.d = d
        self.config = config
        self.device = device
        
        # Set seed for initialization
        g_cpu = torch.Generator()
        g_cpu.manual_seed(config.seed)
        
        # Random projection for SimHash (LSH)
        # We need R_reps * ksim projections
        # G: [d, ksim * R_reps]
        self.G = torch.randn(d, config.ksim * config.R_reps, generator=g_cpu).to(device)
        
        # Random projection for aggregation (if d_proj > 0)
        # W: [d, d_proj]
        if config.d_proj > 0:
            self.W = torch.randn(d, config.d_proj, generator=g_cpu).to(device) / (d ** 0.5)
        else:
            self.W = None
            
        # Final projection (if d_final is set)
        # P: [fde_dim, d_final]
        if config.d_final is not None:
            fde_dim = config.R_reps * (2 ** config.ksim) * (config.d_proj if config.d_proj > 0 else d)
            self.P = torch.randn(fde_dim, config.d_final, generator=g_cpu).to(device) / (fde_dim ** 0.5)
        else:
            self.P = None

    def _compute_hashes(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d]
        # h = sign(x @ G) -> [N, ksim * R]
        
        projs = x @ self.G  # [N, ksim * R]
        bits = (projs > 0).int()
        
        # Reshape to [N, R, ksim]
        bits = bits.view(-1, self.config.R_reps, self.config.ksim)
        
        # Convert bits to integer indices
        # powers of 2: [1, 2, 4, ...]
        powers = 2 ** torch.arange(self.config.ksim, device=self.device)
        bucket_ids = (bits * powers).sum(dim=-1)  # [N, R]
        
        return bucket_ids

    def encode_documents(self, batch_tokens: List[torch.Tensor]) -> torch.Tensor:
        # batch_tokens: list of [Ni, d] tensors
        # Returns: [B, d_final] or [B, fde_dim]
        
        # Vectorized implementation
        # We need to handle variable lengths Ni. 
        # We can pad them to max length in the batch or process them as a single concatenated tensor 
        # and then scatter back, but padding is easier for batch operations if lengths are similar.
        # However, FDE logic is per-token.
        
        # Let's concatenate all tokens: [Total_N, d]
        # Keep track of which doc each token belongs to?
        # Actually, the previous loop was per-document.
        # To vectorize across documents, we can process all tokens at once if we can segment them back.
        
        # But wait, the bucket aggregation is per-document.
        # If we concat all tokens, we get [Total_N, d].
        # We can compute bucket_ids for all tokens at once: [Total_N, R].
        
        # Then we need to aggregate per document.
        # We can use scatter_add with an index that includes the document ID.
        # Global bucket ID = doc_id * (R * num_buckets) + r * num_buckets + bucket_id
        # Then scatter_add into [B * R * num_buckets, val_dim]
        
        B = len(batch_tokens)
        device = self.device
        
        # 1. Concat all tokens
        # sizes: [N1, N2, ..., NB]
        sizes = [t.shape[0] for t in batch_tokens]
        total_tokens = torch.cat(batch_tokens, dim=0) # [Total_N, d]
        
        # Normalize tokens (safety measure)
        # Also cast to G's dtype to prevent fp16/fp32 mismatch if model output is fp16
        total_tokens = normalize_rows(total_tokens).to(dtype=self.G.dtype)
        
        # 2. Project values (if W exists)
        if self.W is not None:
            values = total_tokens @ self.W # [Total_N, d_proj]
            val_dim = self.config.d_proj
        else:
            values = total_tokens
            val_dim = self.d
            
        # 3. Compute bucket IDs for all tokens
        # projs: [Total_N, ksim * R]
        projs = total_tokens @ self.G
        bits = (projs > 0).int()
        bits = bits.view(-1, self.config.R_reps, self.config.ksim) # [Total_N, R, ksim]
        
        powers = 2 ** torch.arange(self.config.ksim, device=device)
        # bucket_ids: [Total_N, R] (values in 0..num_buckets-1)
        bucket_ids = (bits * powers).sum(dim=-1)
        
        num_buckets = 2 ** self.config.ksim
        
        # 4. Create global indices for scatter_add
        # We need to scatter into a tensor of shape [B, R, num_buckets, val_dim]
        # Flattened: [B * R * num_buckets, val_dim]
        
        # Create doc_ids tensor: [Total_N]
        # [0, 0, ..., 1, 1, ..., B-1, ...]
        doc_ids = torch.repeat_interleave(
            torch.arange(B, device=device), 
            torch.tensor(sizes, device=device)
        )
        
        # Expand doc_ids to [Total_N, R]
        doc_ids_expanded = doc_ids.unsqueeze(1).expand(-1, self.config.R_reps)
        
        # R indices: [Total_N, R]
        # [[0, 1, ...], [0, 1, ...], ...]
        r_indices = torch.arange(self.config.R_reps, device=device).unsqueeze(0).expand(total_tokens.shape[0], -1)
        
        # Global index calculation
        # index = doc_id * (R * num_buckets) + r * num_buckets + bucket_id
        # Shape: [Total_N, R]
        global_indices = (
            doc_ids_expanded * (self.config.R_reps * num_buckets) + 
            r_indices * num_buckets + 
            bucket_ids
        )
        
        # Flatten for scatter
        global_indices_flat = global_indices.flatten() # [Total_N * R]
        
        # Values need to be repeated for each R? 
        # No, values are [Total_N, val_dim]. 
        # But we are scattering into R different buckets per token (one for each rep).
        # So we need to repeat values R times?
        # Yes, each token contributes to R buckets (one per rep).
        # values_repeated: [Total_N, R, val_dim]
        values_expanded = values.unsqueeze(1).expand(-1, self.config.R_reps, -1)
        values_flat = values_expanded.reshape(-1, val_dim) # [Total_N * R, val_dim]
        
        # Output tensor
        out_flat = torch.zeros(B * self.config.R_reps * num_buckets, val_dim, device=device)
        
        # Scatter add
        # index needs to be expanded to [Total_N * R, val_dim]
        global_indices_expanded = global_indices_flat.unsqueeze(1).expand(-1, val_dim)
        
        out_flat.scatter_add_(0, global_indices_expanded.long(), values_flat)
        
        # 5. Fill empty clusters
        if self.config.fill_empty_clusters:
            # Count items per bucket
            counts_flat = torch.zeros(B * self.config.R_reps * num_buckets, device=device)
            # We just need to count occurrences of global_indices_flat
            # We can use scatter_add with ones
            ones = torch.ones_like(global_indices_flat, dtype=torch.float)
            counts_flat.scatter_add_(0, global_indices_flat.long(), ones)
            
            # Identify empty buckets
            is_empty = (counts_flat == 0) # [B * R * num_buckets]
            
            if is_empty.any():
                # Calculate mean of non-empty buckets per (doc, rep)
                # Reshape to [B, R, num_buckets]
                # Actually, simpler: calculate mean of all values in the doc?
                # Or mean of non-empty buckets in that specific (doc, rep) group?
                # The paper/MuVERA usually implies filling with the centroid of the document or similar.
                # A simple robust strategy: fill with the mean of the document's values.
                
                # Compute doc means: [B, val_dim]
                # We can compute this from the original values
                # But we have variable lengths.
                # Let's compute sum and count per doc.
                doc_sums = torch.zeros(B, val_dim, device=device)
                doc_sums.index_add_(0, doc_ids, values)
                doc_counts = torch.tensor(sizes, device=device).unsqueeze(1).float()
                doc_means = doc_sums / doc_counts.clamp(min=1.0) # [B, val_dim]
                
                # Expand doc_means to match out_flat
                # [B] -> [B, R, num_buckets] -> [B * R * num_buckets]
                doc_means_expanded = doc_means.unsqueeze(1).unsqueeze(2).expand(-1, self.config.R_reps, num_buckets, -1)
                doc_means_flat = doc_means_expanded.reshape(-1, val_dim)
                
                # Fill empty spots
                # Use torch.where for better performance (avoids nonzero sync)
                is_empty_expanded = is_empty.unsqueeze(-1).expand(-1, val_dim)
                out_flat = torch.where(is_empty_expanded, doc_means_flat, out_flat)

        # 6. L2 Normalize per bucket (optional but recommended for FDE/MuVERA)
        # "Add L2 normalization after scatter_add"
        out_flat = F.normalize(out_flat, p=2, dim=-1)

        # 7. Reshape and Final Projection
        # [B, R, num_buckets, val_dim]
        out_reshaped = out_flat.view(B, self.config.R_reps * num_buckets * val_dim)
        
        if self.P is not None:
            final_fde = out_reshaped @ self.P # [B, d_final]
        else:
            final_fde = out_reshaped
            
        return final_fde

    def build_query_fde(self, q_tokens: torch.Tensor) -> torch.Tensor:
        # q_tokens: [M, d]
        # For query, we typically just use the same encoding logic 
        # or a specific query-side logic. 
        # Here we assume symmetric FDE for simplicity, 
        # but treating the query as a "document" of length M.
        return self.encode_documents([q_tokens])[0]

def build_document_fde_batch(
    D_tokens_list: List[torch.Tensor],
    builder: FDEBuilder,
    sampler: Optional[SamplerConfig] = None,
    batch_size_docs: int = 64,
    r_sub_override: int = 0,
) -> List[torch.Tensor]:
    
    all_fdes = []
    for i in range(0, len(D_tokens_list), batch_size_docs):
        batch = D_tokens_list[i:i+batch_size_docs]
        fdes = builder.encode_documents(batch)
        all_fdes.extend([f for f in fdes])
        
    return all_fdes

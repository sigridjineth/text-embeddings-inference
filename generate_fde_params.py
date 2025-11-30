
import torch
from safetensors.torch import save_file
from fde_pytorch_opt import FDEConfig, FDEBuilder

def generate_params():
    # Configuration matching the default or desired setup
    config = FDEConfig(
        ksim=5,
        d_proj=16,
        R_reps=10,
        d_final=1024,
        seed=42,
        use_mixed_precision=False
    )
    
    # Hidden size for BGE-M3
    hidden_size = 1024 
    
    print(f"Generating FDE params with config: ksim={config.ksim}, d_proj={config.d_proj}, R_reps={config.R_reps}, d_final={config.d_final}")
    
    # Initialize builder
    builder = FDEBuilder(d=hidden_size, config=config, device=torch.device("cpu"))
    
    # Extract matrices
    # G: [R, d, ksim] -> Flatten or keep as is? Candle expects specific shapes.
    # In fde.rs: 
    # g_shape = (hidden_size, config.ksim * config.r_reps);
    # Rust G is [hidden_size, ksim * r_reps]
    # Python G is [R, d, ksim] -> [R, hidden_size, ksim]
    # We need to reshape Python G to match Rust G.
    # Rust G is created as: (0..g_shape.0 * g_shape.1).map(|_| gen_uniform()).collect();
    # And used as: projs = hidden_states.matmul(&self.g)?;
    # hidden_states: [total_tokens, hidden_size]
    # projs: [total_tokens, ksim * r_reps]
    # So G should be [hidden_size, ksim * r_reps]
    
    # Python logic:
    # logits = torch.einsum('md,Rdk->Rmk', X, G)
    # X: [m, d]
    # G: [R, d, k]
    # Result: [R, m, k]
    
    # Rust logic:
    # projs = hidden_states.matmul(&self.g) -> [m, ksim * r_reps]
    # Then reshape to [m, r_reps, ksim] (Wait, let's check fde.rs reshape)
    # let bits = bits.reshape((total_tokens, self.config.r_reps, self.config.ksim))?;
    
    # So if Rust produces [m, r_reps * ksim] and reshapes to [m, r_reps, ksim]
    # It means the last dimension is ksim.
    # So the columns of G in Rust correspond to (r=0, k=0), (r=0, k=1)... (r=1, k=0)...
    
    # Python G is [R, d, k]
    # We want to transform it to [d, R * k] where the R*k dimension is ordered as r0k0, r0k1... r1k0...
    # Python G permute: [d, R, k] -> reshape [d, R*k]
    
    G_py = builder.params.G # [R, d, k]
    G_rust = G_py.permute(1, 0, 2).reshape(hidden_size, -1) # [d, R*k]
    
    # W (Projector S in Python)
    # Rust: w_shape = (hidden_size, config.d_proj);
    # Used as: hidden_states.matmul(w)?
    # Python S: [R, d_proj, d] or None
    # Wait, Rust W is a single matrix [hidden_size, d_proj].
    # Python S is [R, d_proj, d].
    # In Rust implementation I see:
    # let w = if config.d_proj > 0 { ... }
    # It seems Rust implementation uses a SINGLE W for all repetitions?
    # Let's check fde.rs again.
    # "let v = if let Some(w) = &self.w { hidden_states.matmul(w)? } else { hidden_states.clone() };"
    # Yes, Rust uses a global W projection before bucketization/aggregation? 
    # No, wait.
    # In Rust:
    # 1. Normalize tokens
    # 2. Project values (W) -> v
    # 3. Compute bucket IDs (using G)
    # ...
    # 5. Scatter add v to out_flat
    
    # In Python:
    # blocks = _project_block_batched(centroids, S)
    # S is [R, d_proj, d]. It projects centroids [R, B, d] -> [R, B, d_proj].
    # This projection happens AFTER aggregation.
    
    # In Rust:
    # v = hidden_states.matmul(w) -> [total_tokens, d_proj]
    # Then v is aggregated.
    # This means Rust projects BEFORE aggregation.
    # And it uses a single W, not per-repetition S.
    
    # Discrepancy detected!
    # Python: Aggregation -> Projection (per rep)
    # Rust: Projection (global) -> Aggregation
    
    # However, if S is the same for all R, and linear, then Project(Sum(x)) = Sum(Project(x)).
    # But Python S is [R, d_proj, d], so it CAN be different per repetition.
    # Rust W is [hidden_size, d_proj].
    
    # If we want 1:1 equivalence, we must ensure:
    # 1. Python S is same for all R (or Rust supports per-rep W, which it doesn't seem to).
    # 2. Or Rust W is identity (d_proj=0 or d_proj=hidden_size) and Python S is identity.
    # 3. Or we accept they are different architectures.
    
    # The user said: "Python이 만든 G/W/P 파일을 그대로 로딩해서 self.g, self.w, self.p에 집어넣는다."
    # implying we should just load them.
    # But if the shapes/logic don't match, we can't just load them.
    
    # Let's check Python code again.
    # self.S = (torch.randint(0, 2, (R, d_proj, d), ...
    # It generates different S for each R.
    
    # Rust code:
    # let w_shape = (hidden_size, config.d_proj);
    # let w_data: Vec<f32> = ...
    # let w = Tensor::from_vec(...)
    
    # It seems the Rust implementation is a simplified version or a variant.
    # If we want to verify "equivalence", we might need to adjust Rust implementation to match Python, or vice versa.
    # Or maybe I misunderstood Rust code.
    
    # Rust:
    # let v = hidden_states.matmul(w)?
    # ...
    # out_flat = out_flat.index_add(..., &values_flat, 0)?
    # values_flat comes from v.
    
    # So Rust projects tokens first, then aggregates.
    # Python aggregates tokens (centroids), then projects centroids.
    
    # If S is different per rep in Python, Rust cannot match it with a single W.
    # UNLESS we change Rust to use per-rep W.
    # But Rust `v` is [total_tokens, d_proj]. It doesn't have a rep dimension yet.
    # To support per-rep W in Rust, we would need to project `hidden_states` R times, or have `v` be [total_tokens, R, d_proj].
    
    # Given the user wants to verify "TEI FDE vs Python FDE", and explicitly mentioned "G, W, P", maybe they assume the Rust implementation *should* be capable of loading them.
    # If the Rust implementation is "wrong" (simplified), maybe that's part of what needs to be fixed?
    # OR, maybe for the sake of this verification, we force Python to use a single S (broadcasted) and project before aggregation?
    # Actually, `FDEBuilder` in Python is quite flexible.
    
    # Let's look at P (Final Projection).
    # Python: `FinalProjectionStreamer` W is [d_out, d_in].
    # d_in = R * B * d_block.
    # Rust: `p` is [fde_dim, d_final]. fde_dim = R * B * val_dim.
    # This matches! Python W is [d_final, fde_dim]. Rust P is [fde_dim, d_final] (transpose).
    
    # So P is fine. G is fine (just reshape).
    # W (the inner projection) is the problem.
    # Python: S [R, d_proj, d].
    # Rust: W [d, d_proj].
    
    # If we want to verify, we should probably set d_proj = 0 (identity) or d_proj = d in both, to remove this discrepancy for now.
    # Or, we can modify Python generation to use a single S for all R, and check if Project-then-Aggregate is equivalent to Aggregate-then-Project.
    # It is equivalent if S is constant across R.
    # Sum(S @ x) = S @ Sum(x).
    
    # So, for verification, I will:
    # 1. Modify Python generation to use a single S (shared across R).
    # 2. Save this S as W for Rust.
    # 3. In Rust, load W.
    
    # Wait, Python S is [R, d_proj, d]. Rust W is [d, d_proj].
    # So I will generate one S_0 [d_proj, d], replicate it R times for Python S.
    # And save S_0^T [d, d_proj] as W for Rust.
    
    # But wait, `fde_pytorch_opt.py` uses `BatchedParams` which generates random S.
    # I should modify `generate_params` to manually set S in the builder.
    
    # Also, Rust normalizes W by sqrt(d). Python doesn't seem to normalize S in `BatchedParams` (it uses pm1).
    # But `_project_block_batched` just does matmul.
    # If Rust normalizes W, we should save the *normalized* W to the file, or have Rust *not* normalize if loaded from file.
    # I'll assume Rust `FdeModule::new` will just load the tensor as-is and NOT re-normalize if loaded from file.
    
    # Let's prepare the tensors.
    
    # G:
    # Python: [R, d, k]
    # Rust: [d, R*k] (reshaped from [d, R, k])
    
    # W:
    # Python: S [R, d_proj, d].
    # We will enforce S[r] = S[0] for all r.
    # Rust: [d, d_proj].
    # We will save S[0].T.
    
    # P:
    # Python: W [d_final, d_in]
    # Rust: [d_in, d_final]
    # We will save W.T.
    
    tensors = {}
    
    # 1. G
    G_py = builder.params.G # [R, d, k]
    # Permute to [d, R, k] then reshape to [d, R*k]
    G_rust = G_py.permute(1, 0, 2).reshape(hidden_size, -1).contiguous()
    tensors["g"] = G_rust
    
    # 2. W
    # We need to force S to be consistent if we want to match Rust's single W.
    # Or we just take S[0] and tell Python to use it?
    # No, `generate_params` is just generating the file.
    # The verification script `check_bge_m3_fde_equivalence.py` runs Python FDE.
    # It initializes `FDEBuilder` with random seed.
    # We need `check_bge_m3_fde_equivalence.py` to ALSO load these params!
    # The user didn't ask to modify `check_bge_m3_fde_equivalence.py` to load params, but it's implied if we want "1:1 comparison".
    # "Python이 만든 G/W/P 파일을 그대로 로딩해서 self.g, self.w, self.p에 집어넣는다."
    # This applies to TEI.
    # But for Python side, we also need to ensure it uses the SAME params.
    # Since `generate_params.py` creates them using `FDEBuilder`, we can just use the same `FDEBuilder` instance or save/load in Python too.
    
    # Actually, `fde_pytorch_opt.py` has `torch.save(state, "fde_params.pt")` in example.
    # I will make `generate_params.py` save `fde_params.safetensors`.
    # And I will modify `check_bge_m3_fde_equivalence.py` to load it if available?
    # Or just rely on the fact that if I fix the seed in `generate_params.py` and `check_bge_m3_fde_equivalence.py`, they *should* be identical?
    # No, user said "Rust StdRng vs PyTorch RNG" is the problem.
    # So Python side is consistent with itself (PyTorch RNG).
    # The problem is transferring Python params to Rust.
    
    # So:
    # 1. `generate_params.py` uses `FDEBuilder` (PyTorch RNG) to create params.
    # 2. It saves them to `fde_params.safetensors` in a format Rust can understand.
    # 3. Rust loads them.
    # 4. `check_bge_m3_fde_equivalence.py` uses `FDEBuilder` with SAME seed as `generate_params.py`.
    #    Since both use PyTorch RNG, they will generate the same matrices!
    #    So I don't need to modify `check_bge_m3_fde_equivalence.py` to load files, just ensure seeds match.
    
    # BUT, I need to handle the S vs W discrepancy.
    # If I don't change Python's S generation, S will vary per rep.
    # Rust W is single.
    # So they will NEVER match unless I change Python code or Rust code.
    # I will modify `generate_params.py` to FORCE S to be repeated.
    # AND I need to ensure `check_bge_m3_fde_equivalence.py` also uses repeated S.
    # This means I SHOULD modify `fde_pytorch_opt.py` or `check_bge_m3_fde_equivalence.py`.
    
    # Let's modify `fde_pytorch_opt.py` to allow forcing S?
    # Or just monkey-patch it in `generate_params.py` and `check_bge_m3_fde_equivalence.py`.
    
    # Wait, `check_bge_m3_fde_equivalence.py` imports `FDEBuilder`.
    # I can modify `check_bge_m3_fde_equivalence.py` to:
    # 1. Initialize builder.
    # 2. Overwrite builder.params.S with repeated S[0].
    # 3. Save params to safetensors (for Rust).
    # 4. Proceed with encoding.
    
    # This ensures consistency.
    
    # So `generate_params.py` is not strictly needed if I update `check_bge_m3_fde_equivalence.py` to do the saving.
    # I'll update `check_bge_m3_fde_equivalence.py` to perform this "Save Params for TEI" step.
    
    pass

if __name__ == "__main__":
    generate_params()

use candle::{DType, Device, Result, Tensor};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct FdeConfig {
    #[serde(default = "default_ksim")]
    pub ksim: usize,
    #[serde(default = "default_d_proj")]
    pub d_proj: usize,
    #[serde(default = "default_r_reps")]
    pub r_reps: usize,
    #[serde(default = "default_d_final")]
    pub d_final: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_ksim() -> usize { 5 }
fn default_d_proj() -> usize { 16 }
fn default_r_reps() -> usize { 10 }
fn default_d_final() -> usize { 1024 }
fn default_seed() -> u64 { 42 }

impl Default for FdeConfig {
    fn default() -> Self {
        Self {
            ksim: default_ksim(),
            d_proj: default_d_proj(),
            r_reps: default_r_reps(),
            d_final: default_d_final(),
            seed: default_seed(),
        }
    }
}

impl FdeConfig {
    pub fn from_env() -> Result<Self> {
        let ksim = std::env::var("FDE_KSIM").ok().and_then(|v| v.parse().ok()).unwrap_or_else(default_ksim);
        let d_proj = std::env::var("FDE_D_PROJ").ok().and_then(|v| v.parse().ok()).unwrap_or_else(default_d_proj);
        let r_reps = std::env::var("FDE_R_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(default_r_reps);
        let d_final = std::env::var("FDE_D_FINAL").ok().and_then(|v| v.parse().ok()).unwrap_or_else(default_d_final);
        let seed = std::env::var("FDE_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or_else(default_seed);
        
        Ok(Self {
            ksim,
            d_proj,
            r_reps,
            d_final,
            seed,
        })
    }
}

pub struct FdeModule {
    g: Tensor,
    w: Option<Tensor>,
    p: Option<Tensor>,
    config: FdeConfig,
    device: Device,
}

impl FdeModule {
    pub fn new(config: FdeConfig, hidden_size: usize, device: &Device) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(config.seed);

        // Helper for Uniform[-1, 1]
        #[allow(deprecated)]
        let mut gen_uniform = || (2.0 * rng.gen::<f32>() - 1.0);

        // G: [hidden_size, ksim * r_reps]
        let g_shape = (hidden_size, config.ksim * config.r_reps);
        let g_data: Vec<f32> = (0..g_shape.0 * g_shape.1).map(|_| gen_uniform()).collect();
        let g = Tensor::from_vec(g_data, g_shape, device)?.to_dtype(DType::F16)?;

        // W: [hidden_size, d_proj]
        let w = if config.d_proj > 0 {
            let w_shape = (hidden_size, config.d_proj);
            let w_data: Vec<f32> = (0..w_shape.0 * w_shape.1).map(|_| gen_uniform()).collect();
            let w = Tensor::from_vec(w_data, w_shape, device)?.to_dtype(DType::F16)?;
            // Normalize by sqrt(d)
            Some((w / (hidden_size as f64).sqrt())?)
        } else {
            None
        };

        // P: [fde_dim, d_final]
        let p = if config.d_final > 0 {
            let val_dim = if config.d_proj > 0 { config.d_proj } else { hidden_size };
            let fde_dim = config.r_reps * (1 << config.ksim) * val_dim;
            let p_shape = (fde_dim, config.d_final);
            let p_data: Vec<f32> = (0..p_shape.0 * p_shape.1).map(|_| gen_uniform()).collect();
            let p = Tensor::from_vec(p_data, p_shape, device)?.to_dtype(DType::F16)?;
            // Normalize by sqrt(fde_dim)
            Some((p / (fde_dim as f64).sqrt())?)
        } else {
            None
        };

        Ok(Self {
            g,
            w,
            p,
            config,
            device: device.clone(),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, cumulative_seq_lengths: &[u32]) -> Result<Tensor> {
        // hidden_states: [total_tokens, hidden_size]
        // cumulative_seq_lengths: [batch_size + 1]
        
        let total_tokens = hidden_states.dim(0)?;
        let batch_size = cumulative_seq_lengths.len() - 1;
        let num_buckets = 1 << self.config.ksim;
        let val_dim = if self.config.d_proj > 0 { self.config.d_proj } else { hidden_states.dim(1)? };

        // Output tensor: [batch_size * r_reps * num_buckets, val_dim]
        let out_size = batch_size * self.config.r_reps * num_buckets;
        let mut out_flat = Tensor::zeros((out_size, val_dim), DType::F16, &self.device)?;
        
        // We need these for filling empty clusters
        let mut global_indices_flat = None;
        let mut values = None;
        let mut doc_ids = None;

        if total_tokens > 0 {
            // 1. Normalize tokens
            let hidden_states = normalize_rows(hidden_states)?;

            // 2. Project values (W)
            let v = if let Some(w) = &self.w {
                hidden_states.matmul(w)?
            } else {
                hidden_states.clone()
            };
            
            // 3. Compute bucket IDs
            // projs: [total_tokens, ksim * r_reps]
            let projs = hidden_states.matmul(&self.g)?;
            let bits = projs.gt(&projs.zeros_like()?)?.to_dtype(DType::U32)?; // [total_tokens, ksim * r_reps]
            
            // Reshape bits to [total_tokens, r_reps, ksim]
            let bits = bits.reshape((total_tokens, self.config.r_reps, self.config.ksim))?;

            // Powers of 2: [1, 2, 4, ...]
            let powers: Vec<u32> = (0..self.config.ksim).map(|i| 1 << i).collect();
            let powers = Tensor::from_vec(powers, (self.config.ksim,), &self.device)?;
            
            // bucket_ids: [total_tokens, r_reps]
            // Summing in F32 to ensure support, then casting to I64 for indexing
            let bits_f = bits.to_dtype(DType::F32)?;
            let powers_f = powers.to_dtype(DType::F32)?;
            let bucket_ids = (bits_f.broadcast_mul(&powers_f)?.sum_keepdim(2)?.squeeze(2))?.to_dtype(DType::I64)?;

            // 4. Create global indices for scatter_add
            // We need doc_ids for each token.
            // cumulative_seq_lengths gives us the boundaries.
            // We can construct doc_ids tensor.
            let mut doc_ids_vec = Vec::with_capacity(total_tokens);
            for i in 0..batch_size {
                let start = cumulative_seq_lengths[i] as usize;
                let end = cumulative_seq_lengths[i+1] as usize;
                for _ in start..end {
                    doc_ids_vec.push(i as u32);
                }
            }
            let d_ids = Tensor::from_vec(doc_ids_vec, (total_tokens,), &self.device)?.to_dtype(DType::I64)?;
            
            // Expand doc_ids: [total_tokens, r_reps]
            let doc_ids_expanded = d_ids.unsqueeze(1)?.broadcast_as((total_tokens, self.config.r_reps))?;

            // R indices: [total_tokens, r_reps]
            let r_indices: Vec<u32> = (0..self.config.r_reps as u32).collect();
            let r_indices = Tensor::from_vec(r_indices, (self.config.r_reps,), &self.device)?.to_dtype(DType::I64)?;
            let r_indices = r_indices.unsqueeze(0)?.broadcast_as((total_tokens, self.config.r_reps))?;

            // Global index: doc_id * (R * num_buckets) + r * num_buckets + bucket_id
            // Shape: [total_tokens, r_reps]
            let stride_doc = (self.config.r_reps * num_buckets) as u32;
            let stride_r = num_buckets as u32;
            
            // Convert strides to tensors for broadcasting (I64 arithmetic)
            let stride_doc_t = Tensor::new(stride_doc as i64, &self.device)?;
            let stride_r_t = Tensor::new(stride_r as i64, &self.device)?;
            
            let global_indices = doc_ids_expanded.broadcast_mul(&stride_doc_t)?
                .add(&r_indices.broadcast_mul(&stride_r_t)?)?
                .add(&bucket_ids)?;
            
            // Flatten indices: [total_tokens * r_reps]
            let g_indices_flat = global_indices.flatten_all()?.to_dtype(DType::I64)?;

            // Values need to be repeated for each rep?
            // values: [total_tokens, val_dim]
            // We need [total_tokens, r_reps, val_dim] -> [total_tokens * r_reps, val_dim]
            let values_expanded = v.unsqueeze(1)?.broadcast_as((total_tokens, self.config.r_reps, val_dim))?;
            let values_flat = values_expanded.reshape((total_tokens * self.config.r_reps, val_dim))?;

            // Scatter add
            // Candle's index_add adds `source` to `self` at `indices`.
            // `self.index_add(dim, index, source)`
            // dim=0. index=global_indices_flat. source=values_flat.
            out_flat = out_flat.index_add(&g_indices_flat, &values_flat, 0)?;
            
            global_indices_flat = Some(g_indices_flat);
            values = Some(v);
            doc_ids = Some(d_ids);
        }

        // 5. Fill empty clusters (Simple mean strategy)
        // Count items per bucket
        let counts_flat = if let Some(indices) = &global_indices_flat {
             let ones = Tensor::ones((indices.dim(0)?,), DType::F16, &self.device)?;
             let mut counts = Tensor::zeros((out_size,), DType::F16, &self.device)?;
             counts = counts.index_add(indices, &ones, 0)?;
             counts
        } else {
             Tensor::zeros((out_size,), DType::F16, &self.device)?
        };

        let is_empty = counts_flat.eq(&counts_flat.zeros_like()?)?; // [out_size]

        // Compute doc means
        let doc_means = if let (Some(v), Some(d_ids)) = (&values, &doc_ids) {
            let mut doc_sums = Tensor::zeros((batch_size, val_dim), DType::F16, &self.device)?;
            doc_sums = doc_sums.index_add(d_ids, v, 0)?;
            
            let mut doc_lens_vec = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let len = (cumulative_seq_lengths[i+1] - cumulative_seq_lengths[i]) as f64;
                doc_lens_vec.push(len.max(1.0));
            }
            let doc_lens = Tensor::from_vec(doc_lens_vec, (batch_size, 1), &self.device)?.to_dtype(DType::F16)?;
            (doc_sums / doc_lens)? // [batch_size, val_dim]
        } else {
            Tensor::zeros((batch_size, val_dim), DType::F16, &self.device)?
        };

        // Expand doc_means to [out_size, val_dim]
        // out_size = batch_size * r_reps * num_buckets
        // We need to repeat each doc_mean (r_reps * num_buckets) times.
        let doc_means_expanded = doc_means.unsqueeze(1)?
            .broadcast_as((batch_size, self.config.r_reps * num_buckets, val_dim))?
            .reshape((out_size, val_dim))?;

        // Fill empty
        // where_cond(condition, on_true, on_false)
        // condition must be broadcastable. is_empty is [out_size].
        let is_empty_expanded = is_empty.unsqueeze(1)?.broadcast_as((out_size, val_dim))?;
        let out_flat = is_empty_expanded.where_cond(&doc_means_expanded, &out_flat)?;

        // 6. L2 Normalize per bucket
        let out_flat = normalize_rows(&out_flat)?;

        // 7. Reshape and Final Projection
        // [batch_size, r_reps * num_buckets * val_dim]
        let out_reshaped = out_flat.reshape((batch_size, self.config.r_reps * num_buckets * val_dim))?;

        let final_fde = if let Some(p) = &self.p {
            out_reshaped.matmul(p)?
        } else {
            out_reshaped
        };

        Ok(final_fde)
    }
}

fn normalize_rows(x: &Tensor) -> Result<Tensor> {
    let norm = (x.sqr()?.sum_keepdim(1)? + 1e-12)?.sqrt()?;
    x.broadcast_div(&norm)
}

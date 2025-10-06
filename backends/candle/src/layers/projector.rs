//! Jina v3 Reranker MLP Projector
//!
//! Architecture: Linear(hidden_size → hidden_size/2, bias=False) → ReLU → Linear(hidden_size/2 → 512, bias=False)

use candle::{DType, Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug)]
pub struct Projector {
    w1: Tensor,
    w2: Tensor,
}

impl Projector {
    /// Load projector weights from VarBuilder
    ///
    /// # Arguments
    ///
    /// * `vb` - VarBuilder (should be set to model dtype via vb.set_dtype())
    /// * `hidden_size` - Model hidden size (e.g., 1024 for Qwen3)
    pub fn load(vb: VarBuilder, hidden_size: usize) -> Result<Self> {
        let latent_size = hidden_size / 2; // modeling.py: hidden_size → hidden_size/2 → 512

        // VarBuilder paths map to safetensors keys:
        // VarBuilder is already scoped to "projector" from lib.rs:242
        // So vb.pp("0") → "projector.0.weight" (not vb.pp("projector").pp("0"))
        let w1 = vb.pp("0").get((latent_size, hidden_size), "weight")?;
        let w2 = vb.pp("2").get((512, latent_size), "weight")?;

        // CRITICAL FIX: ensure projector weights are materialised as F32 on the target device.
        let w1 = w1
            .to_device(vb.device())?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let w2 = w2
            .to_device(vb.device())?
            .to_dtype(DType::F32)?
            .contiguous()?;

        // Verify projector has no bias (modeling.py: bias=False)
        // Check existence using contains_tensor to avoid partial loads.
        if vb.pp("0").contains_tensor("bias") || vb.pp("2").contains_tensor("bias") {
            candle::bail!(
                "Projector must be bias-free (bias=False per Jina v3 spec). \
                 This model may not be compatible. Verify weights or use --reranker-mode pairwise"
            );
        }

        Ok(Self { w1, w2 })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        // All projector math happens in F32 to satisfy Candle's matmul requirements.
        let hidden = if hidden.dtype() == DType::F32 {
            hidden.clone()
        } else {
            hidden.to_dtype(DType::F32)?
        };

        let hidden = hidden.contiguous()?;
        let h1 = hidden.matmul(&self.w1.t()?)?.relu()?;
        h1.matmul(&self.w2.t()?)
    }
}

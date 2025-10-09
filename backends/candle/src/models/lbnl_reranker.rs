//! LBNL Reranker model: Qwen3 + MLP Projector
use crate::layers::projector::Projector;
use crate::models::{qwen3::Qwen3Model, Model};
use candle::{DType, Device, IndexOp, Result as CResult, Tensor};
use candle_nn::VarBuilder;
use text_embeddings_backend_core::{
    Backend, BackendError, Batch, ListwiseBlockInput, ListwiseBlockOutput,
};

pub struct LbnlReranker {
    qwen3: Qwen3Model,
    projector: Projector,
    device: Device,
}

impl LbnlReranker {
    pub fn new(
        vb: VarBuilder,
        qwen3: Qwen3Model,
        device: Device,
        hidden_size: usize,
        _dtype: DType, // Kept for API compatibility but unused (we always convert to F32)
    ) -> CResult<Self> {
        // Load projector weights from VarBuilder
        // Note: We always convert hidden states to F32 before projector to avoid dtype issues
        let projector = Projector::load(vb, hidden_size)?;
        Ok(Self {
            qwen3,
            projector,
            device,
        })
    }

    pub fn forward(&self, input: &ListwiseBlockInput) -> CResult<ListwiseBlockOutput> {
        let t = input.input_ids.len();

        let ids = Tensor::from_vec(input.input_ids.clone(), (1, t), &self.device)?;
        let mask = Tensor::from_vec(input.attention_mask.clone(), (1, t), &self.device)?;

        // Use forward_with_tensors for hidden states extraction
        let hs = self.qwen3.forward_with_tensors(&ids, &mask)?;

        // Ensure hidden states are F32 before projector
        let hs = if hs.dtype() != DType::F32 {
            tracing::warn!(
                "Hidden states dtype is {:?}, converting to F32 for projector.",
                hs.dtype()
            );
            hs.to_dtype(DType::F32)?
        } else {
            hs
        };

        // Find special token positions
        let mut rerank_pos = None;
        let mut doc_token_positions = Vec::with_capacity(input.doc_count);

        for (i, &tid) in input.input_ids.iter().enumerate() {
            if tid == input.embed_token_id {
                // Doc embedding is from the token *before* the marker
                doc_token_positions.push(i.saturating_sub(1));
            }
            if tid == input.rerank_token_id {
                // Query embedding is also from the token *before* the marker
                rerank_pos = Some(i.saturating_sub(1));
            }
        }
        let qpos = rerank_pos.ok_or_else(|| candle::Error::Msg("No rerank token found".into()))?;

        // Extract hidden states at positions → F32 [1, H]
        // Note: hs is already F32 from conversion above, but we keep .to_dtype for safety
        let hq = hs.i((0, qpos, ..))?.to_dtype(DType::F32)?.unsqueeze(0)?;

        // Process documents: extract and pass through projector
        let mut doc_embs = Vec::with_capacity(doc_token_positions.len());
        for &p in &doc_token_positions {
            // Extract and ensure F32 dtype before projector
            let hd = hs.i((0, p, ..))?.to_dtype(DType::F32)?.unsqueeze(0)?;

            // Pass through projector (projector weights should be F32 compatible)
            let zd = self.projector.forward(&hd)?;

            // Convert output to F32 if needed and extract as Vec
            let zd_f32 = if zd.dtype() != DType::F32 {
                zd.to_dtype(DType::F32)?
            } else {
                zd
            };
            doc_embs.push(zd_f32.to_vec2::<f32>()?.remove(0));
        }

        // Process query: same approach
        let zq = self.projector.forward(&hq)?;
        let zq_f32 = if zq.dtype() != DType::F32 {
            zq.to_dtype(DType::F32)?
        } else {
            zq
        };
        let zq_vec = zq_f32.to_vec2::<f32>()?.remove(0);

        // Important normalization policy (modeling.py parity):
        // - Projector outputs are returned WITHOUT L2 normalization
        // - Router handler performs normalization inside cosine_similarity()
        // - This matches Python reference where normalize() is called in compute_scores()
        // - Normalizing here would cause double normalization!

        Ok(ListwiseBlockOutput {
            query_embedding: zq_vec,
            doc_embeddings: doc_embs,
        })
    }
}

// Implement Model trait for Candle backend integration
impl Model for LbnlReranker {
    fn is_padded(&self) -> bool {
        true // Qwen3 uses left padding
    }

    // LBNL reranker doesn't support standard embedding
    fn embed(&self, _batch: Batch) -> candle::Result<(Option<Tensor>, Option<Tensor>)> {
        candle::bail!("LBNL reranker only supports listwise reranking, not standard embedding")
    }

    // LBNL reranker doesn't support pairwise prediction
    fn predict(&self, _batch: Batch) -> candle::Result<Tensor> {
        candle::bail!("LBNL reranker only supports listwise reranking, not pairwise prediction")
    }

    fn embed_listwise_block(
        &self,
        input: ListwiseBlockInput,
    ) -> candle::Result<ListwiseBlockOutput> {
        self.forward(&input)
    }
}

// Implement Backend trait (not a separate ListwiseBackend)
// This allows dispatch via Box<dyn Backend> without downcasting
impl Backend for LbnlReranker {
    fn health(&self) -> Result<(), BackendError> {
        Ok(()) // Model loaded successfully
    }

    fn is_padded(&self) -> bool {
        true // Qwen3 uses left padding
    }

    fn embed(
        &self,
        _batch: Batch,
    ) -> Result<text_embeddings_backend_core::Embeddings, BackendError> {
        Err(BackendError::Inference(
            "LBNL reranker only supports embed_listwise_block, not standard embedding".into(),
        ))
    }

    fn predict(
        &self,
        _batch: Batch,
    ) -> Result<text_embeddings_backend_core::Predictions, BackendError> {
        Err(BackendError::Inference(
            "LBNL reranker only supports embed_listwise_block, not pairwise prediction".into(),
        ))
    }

    // Override default implementation to provide listwise support
    fn embed_listwise_block(
        &self,
        input: ListwiseBlockInput,
    ) -> Result<ListwiseBlockOutput, BackendError> {
        self.forward(&input)
            .map_err(|e| BackendError::Inference(e.to_string()))
    }
}

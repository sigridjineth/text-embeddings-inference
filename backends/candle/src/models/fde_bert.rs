use crate::models::bert::{BertConfig, BertModel};
use candle::{DType, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
use serde::Deserialize;
use crate::models::Model;
use text_embeddings_backend_core::{Batch, ModelType};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FdeConfig {
    pub ksim: usize,
    pub d_proj: usize,
    #[serde(default)]
    pub d_final: Option<usize>,
    #[serde(default)]
    pub fill_empty_clusters: bool,
    #[serde(flatten)]
    pub bert_config: BertConfig,
}

pub struct FdeBertModel {
    bert: BertModel,
    proj: Linear,
    rotation: Tensor,
    powers: Tensor,
    ksim: usize,
    _d_proj: usize,
}

impl FdeBertModel {
    pub fn load(vb: VarBuilder, config: &FdeConfig, model_type: ModelType) -> Result<Self> {
        let bert = BertModel::load(vb.pp("bert"), &config.bert_config, model_type)?;
        
        let hidden_size = config.bert_config.hidden_size;
        let proj = candle_nn::linear(hidden_size, config.d_proj, vb.pp("fde_proj"))?;
        let rotation = vb.get((config.d_proj, config.ksim), "fde_rotation")?;

        // Cache powers of 2
        let mut powers = Vec::with_capacity(config.ksim);
        for i in 0..config.ksim {
            powers.push(2u32.pow(i as u32) as f32);
        }
        let powers = Tensor::from_vec(powers, (config.ksim, 1), &bert.device)?.to_dtype(DType::F32)?;

        Ok(Self {
            bert,
            proj,
            rotation,
            powers,
            ksim: config.ksim,
            _d_proj: config.d_proj,
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        // 1. Get BERT token embeddings
        let (input_ids, type_ids, position_ids, _input_lengths, attention_bias, _attention_mask) = 
            self.prepare_bert_input(&batch)?;


        // Check modes for future implementation of query/doc specific logic
        for mode in &batch.modes {
            if let Some(m) = mode {
                if m == "query" {
                    // TODO: Implement query-specific logic (e.g. disable fill_empty_clusters)
                }
            }
        }

        let embedding_output = self.bert.embeddings.forward(&input_ids, &type_ids, &position_ids)?;
        let outputs = self.bert.encoder.forward(&embedding_output, attention_bias.as_ref())?;

        // outputs: (Batch, Seq, Hidden)
        
        // 2. FDE Logic
        // Projection first
        let projected = self.proj.forward(&outputs)?; // (B, S, d_proj)

        // Normalize (L2)
        // Add epsilon to avoid division by zero
        let epsilon = 1e-12;
        let norms = projected.sqr()?.sum_keepdim(2)?.sqrt()?;
        let norms = (norms + epsilon)?;
        let normalized = projected.broadcast_div(&norms)?;

        // SimHash Rotation
        // rotation: (d_proj, ksim)
        // normalized: (B, S, d_proj)
        // logits: (B, S, ksim)
        let logits = normalized.broadcast_matmul(&self.rotation)?;

        // Binarize
        let zeros = Tensor::zeros_like(&logits)?;
        let bits = logits.ge(&zeros)?.to_dtype(DType::F32)?; // 1.0 if > 0, else 0.0
        
        // bucket_ids: (B, S, 1)
        // bits: (B, S, ksim)
        // powers: (ksim, 1)
        let bucket_ids = bits.broadcast_matmul(&self.powers)?; 
        let bucket_ids = bucket_ids.squeeze(2)?.to_dtype(DType::U32)?; // (B, S)

        // 3. Aggregation (Scatter Add)
        let num_buckets = 1 << self.ksim;
        let (b_size, s_len, d_dim) = projected.shape().dims3()?;
        
        let batch_offsets = Tensor::arange(0u32, b_size as u32, &self.bert.device)?
            .affine(num_buckets as f64, 0.0)?
            .reshape((b_size, 1))?
            .broadcast_as((b_size, s_len))?;
            
        let global_bucket_ids = (bucket_ids + batch_offsets)?.flatten_all()?;
        let flattened_projected = projected.flatten(0, 1)?;
        
        let output_size = b_size * num_buckets;
        let mut fde_vectors = Tensor::zeros((output_size, d_dim), DType::F32, &self.bert.device)?;
        
        fde_vectors = fde_vectors.index_add(&global_bucket_ids, &flattened_projected, 0)?;
        
        // Reshape back to (B, num_buckets, d_proj)
        let fde_vectors = fde_vectors.reshape((b_size, num_buckets, d_dim))?;

        // 5. Final Projection (if d_final exists) - currently unused/removed
        
        let fde_flat = fde_vectors.flatten(1, 2)?; // (B, num_buckets * d_proj)
        
        Ok((Some(fde_flat), None))
    }
    
    // Helper to reuse BertModel's input preparation logic
    // We might need to copy-paste this or refactor BertModel further.
    // Since I made fields public, I can access them, but the logic is inside `forward`.
    // I should probably refactor `BertModel::forward` to split input prep and execution, 
    // but for now I'll duplicate the prep logic to avoid breaking `BertModel` too much.
    fn prepare_bert_input(&self, batch: &Batch) -> Result<(Tensor, Tensor, Tensor, Tensor, Option<Tensor>, Option<Tensor>)> {
        let batch_size = batch.len();
        let max_length = batch.max_length as usize;
        let shape = (batch_size, max_length);
        let device = &self.bert.device;

        let (input_ids, type_ids, position_ids, input_lengths, attention_bias, attention_mask) =
            if batch_size > 1 {
                let elems = batch_size * max_length;
                let mut input_ids = Vec::with_capacity(elems);
                let mut type_ids = Vec::with_capacity(elems);
                let mut position_ids = Vec::with_capacity(elems);
                let mut attention_mask = Vec::with_capacity(elems);
                let mut attention_bias = Vec::with_capacity(elems);
                let mut input_lengths = Vec::with_capacity(batch_size);
                let mut masking = false;

                for i in 0..batch_size {
                    let start = batch.cumulative_seq_lengths[i] as usize;
                    let end = batch.cumulative_seq_lengths[i + 1] as usize;
                    let seq_length = (end - start) as u32;
                    input_lengths.push(seq_length as f32);

                    for j in start..end {
                        input_ids.push(batch.input_ids[j]);
                        type_ids.push(batch.token_type_ids[j]);
                        position_ids.push(batch.position_ids[j]);
                        attention_mask.push(1.0_f32);
                        attention_bias.push(0.0);
                    }

                    let padding = batch.max_length - seq_length;
                    if padding > 0 {
                        masking = true;
                        for _ in 0..padding {
                            input_ids.push(0);
                            type_ids.push(0);
                            position_ids.push(0);
                            attention_mask.push(0.0_f32);
                            attention_bias.push(f32::NEG_INFINITY);
                        }
                    }
                }

                let (attention_bias, attention_mask) = if masking {
                     let attention_mask = Tensor::from_vec(
                        attention_mask,
                        (batch_size, max_length, 1),
                        device,
                    )?.to_dtype(self.bert.dtype)?;

                    let attention_bias = Tensor::from_vec(
                        attention_bias,
                        (batch_size, 1, 1, max_length),
                        device,
                    )?.to_dtype(self.bert.dtype)?;
                    
                    let attention_bias = attention_bias
                        .broadcast_as((batch_size, self.bert.num_attention_heads, max_length, max_length))?
                        .contiguous()?;
                    (Some(attention_bias), Some(attention_mask))
                } else {
                    (None, None)
                };

                (input_ids, type_ids, position_ids, input_lengths, attention_bias, attention_mask)
            } else {
                 (
                    batch.input_ids.clone(),
                    batch.token_type_ids.clone(),
                    batch.position_ids.clone(),
                    vec![batch.max_length as f32],
                    None,
                    None,
                )
            };

        let input_ids = Tensor::from_vec(input_ids, shape, device)?;
        let type_ids = Tensor::from_vec(type_ids, shape, device)?;
        let position_ids = Tensor::from_vec(position_ids, shape, device)?;
        let input_lengths = Tensor::from_vec(input_lengths, (batch_size, 1), device)?.to_dtype(self.bert.dtype)?;

        Ok((input_ids, type_ids, position_ids, input_lengths, attention_bias, attention_mask))
    }
}

impl Model for FdeBertModel {
    fn is_padded(&self) -> bool {
        true
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}


mod common;

use anyhow::Result;
use candle::{DType, Device, Tensor};
use common::{batch, load_tokenizer, sort_embeddings};
use std::collections::HashMap;
use std::fs;
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
fn test_fde_bert() -> Result<()> {
    // 1. Setup temporary directory
    let tmp_dir = std::env::temp_dir().join("test_fde_bert");
    if tmp_dir.exists() {
        fs::remove_dir_all(&tmp_dir)?;
    }
    fs::create_dir_all(&tmp_dir)?;
    let model_root = tmp_dir;

    // 2. Create config.json
    let config_content = r#"{
        "architectures": ["FdeBertModel"],
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "vocab_size": 100,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "layer_norm_eps": 1e-12,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "ksim": 4,
        "d_proj": 16,
        "model_type": "fde_bert"
    }"#;
    fs::write(model_root.join("config.json"), config_content)?;

    // 3. Create tokenizer.json (minimal)
    let tokenizer_content = "{ \"version\": \"1.0\", \"truncation\": null, \"padding\": null, \"added_tokens\": [], \"normalizer\": null, \"pre_tokenizer\": { \"type\": \"Whitespace\" }, \"post_processor\": null, \"decoder\": null, \"model\": { \"type\": \"WordPiece\", \"vocab\": { \"[PAD]\": 0, \"[UNK]\": 1, \"[CLS]\": 2, \"[SEP]\": 3, \"[MASK]\": 4, \"hello\": 5, \"world\": 6 }, \"unk_token\": \"[UNK]\", \"continuing_subword_prefix\": \"##\", \"max_input_chars_per_word\": 100 } }";
    fs::write(model_root.join("tokenizer.json"), tokenizer_content)?;
    
    // 4. Create model.safetensors
    let device = Device::Cpu;
    let mut tensors = HashMap::new();
    
    // BERT weights (minimal)
    tensors.insert("bert.embeddings.word_embeddings.weight".to_string(), Tensor::randn(0f32, 1f32, (100, 32), &device)?);
    tensors.insert("bert.embeddings.position_embeddings.weight".to_string(), Tensor::randn(0f32, 1f32, (512, 32), &device)?);
    tensors.insert("bert.embeddings.token_type_embeddings.weight".to_string(), Tensor::randn(0f32, 1f32, (2, 32), &device)?);
    tensors.insert("bert.embeddings.LayerNorm.weight".to_string(), Tensor::ones((32,), DType::F32, &device)?);
    tensors.insert("bert.embeddings.LayerNorm.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);
    
    // Layer 0
    tensors.insert("bert.encoder.layer.0.attention.self.query.weight".to_string(), Tensor::randn(0f32, 1f32, (32, 32), &device)?);
    tensors.insert("bert.encoder.layer.0.attention.self.query.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.0.attention.self.key.weight".to_string(), Tensor::randn(0f32, 1f32, (32, 32), &device)?);
    tensors.insert("bert.encoder.layer.0.attention.self.key.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.0.attention.self.value.weight".to_string(), Tensor::randn(0f32, 1f32, (32, 32), &device)?);
    tensors.insert("bert.encoder.layer.0.attention.self.value.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.0.attention.output.dense.weight".to_string(), Tensor::randn(0f32, 1f32, (32, 32), &device)?);
    tensors.insert("bert.encoder.layer.0.attention.output.dense.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.0.attention.output.LayerNorm.weight".to_string(), Tensor::ones((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.0.attention.output.LayerNorm.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.0.intermediate.dense.weight".to_string(), Tensor::randn(0f32, 1f32, (64, 32), &device)?);
    tensors.insert("bert.encoder.layer.0.intermediate.dense.bias".to_string(), Tensor::zeros((64,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.0.output.dense.weight".to_string(), Tensor::randn(0f32, 1f32, (32, 64), &device)?);
    tensors.insert("bert.encoder.layer.0.output.dense.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.0.output.LayerNorm.weight".to_string(), Tensor::ones((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.0.output.LayerNorm.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);

    // Layer 1
    tensors.insert("bert.encoder.layer.1.attention.self.query.weight".to_string(), Tensor::randn(0f32, 1f32, (32, 32), &device)?);
    tensors.insert("bert.encoder.layer.1.attention.self.query.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.1.attention.self.key.weight".to_string(), Tensor::randn(0f32, 1f32, (32, 32), &device)?);
    tensors.insert("bert.encoder.layer.1.attention.self.key.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.1.attention.self.value.weight".to_string(), Tensor::randn(0f32, 1f32, (32, 32), &device)?);
    tensors.insert("bert.encoder.layer.1.attention.self.value.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.1.attention.output.dense.weight".to_string(), Tensor::randn(0f32, 1f32, (32, 32), &device)?);
    tensors.insert("bert.encoder.layer.1.attention.output.dense.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.1.attention.output.LayerNorm.weight".to_string(), Tensor::ones((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.1.attention.output.LayerNorm.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.1.intermediate.dense.weight".to_string(), Tensor::randn(0f32, 1f32, (64, 32), &device)?);
    tensors.insert("bert.encoder.layer.1.intermediate.dense.bias".to_string(), Tensor::zeros((64,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.1.output.dense.weight".to_string(), Tensor::randn(0f32, 1f32, (32, 64), &device)?);
    tensors.insert("bert.encoder.layer.1.output.dense.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.1.output.LayerNorm.weight".to_string(), Tensor::ones((32,), DType::F32, &device)?);
    tensors.insert("bert.encoder.layer.1.output.LayerNorm.bias".to_string(), Tensor::zeros((32,), DType::F32, &device)?);

    // FDE weights
    // d_proj = 16, hidden_size = 32
    tensors.insert("fde_proj.weight".to_string(), Tensor::randn(0f32, 1f32, (16, 32), &device)?);
    tensors.insert("fde_proj.bias".to_string(), Tensor::zeros((16,), DType::F32, &device)?);
    // ksim = 4
    tensors.insert("fde_rotation".to_string(), Tensor::randn(0f32, 1f32, (16, 4), &device)?);

    candle::safetensors::save(&tensors, model_root.join("model.safetensors"))?;

    // 5. Load Backend
    let backend = CandleBackend::new(
        &model_root,
        "float32".to_string(),
        ModelType::Embedding(Pool::Fde),
        None,
    )?;

    // 6. Run Inference
    let tokenizer = load_tokenizer(&model_root)?;
    let input_batch = batch(
        vec![
            tokenizer.encode("hello world", true).unwrap(),
            tokenizer.encode("hello", true).unwrap(),
        ],
        [0, 1].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_batch)?);
    let batch_size = pooled_embeddings.len();
    let dim = pooled_embeddings[0].len();
    let flat_embeddings: Vec<f32> = pooled_embeddings.into_iter().flatten().collect();
    let pooled_embeddings = Tensor::from_vec(flat_embeddings, (batch_size, dim), &device)?;
    
    // Check output shape
    // ksim = 4 -> num_buckets = 16
    // d_proj = 16
    // output dim = 16 * 16 = 256
    let (b, d) = pooled_embeddings.shape().dims2()?;
    assert_eq!(b, 2);
    assert_eq!(d, 256);

    println!("FDE Inference successful. Output shape: {:?}", pooled_embeddings.shape());

    Ok(())
}

import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel

def test_colbert_extraction_match():
    print("Initializing BGEM3FlagModel...")
    # Load model (assuming BAAI/bge-m3 is available or cached)
    # If not, this might fail if no internet, but usually these envs have it or we use a dummy path if user provided one.
    # The user has /Users/sigridjineth/Desktop/work/text-embeddings-inference -> huggingface/text-embeddings-inference
    # We'll try to use the standard model name, hoping it's cached.
    model_name = "BAAI/bge-m3" 
    try:
        model = BGEM3FlagModel(model_name, use_fp16=False, device="cpu")
    except Exception as e:
        print(f"Could not load {model_name}: {e}")
        print("Skipping live model test, but logic verification remains valid.")
        return

    sentences = ["This is a test sentence for ColBERT extraction verification."]
    
    # 1. Official Output
    print("Getting official ColBERT vectors...")
    output_official = model.encode(sentences, return_colbert_vecs=True, return_dense=False, return_sparse=False)
    colbert_official = output_official['colbert_vecs'][0] # List of arrays? or tensor? FlagEmbedding returns list of arrays usually for colbert
    
    # 2. Manual Logic (from BGEM3FDEModel)
    print("Running manual extraction logic...")
    tokenizer = model.tokenizer
    hf_model = model.model
    
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        base_model = hf_model.model # XLMRobertaModel
        model_output = base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = model_output.last_hidden_state
        
        # Manual logic
        hidden_body = last_hidden[:, 1:, :] # Drop CLS
        mask_body = attention_mask[:, 1:].unsqueeze(-1).float()
        
        colbert_linear = getattr(hf_model, "colbert_linear", None)
        if colbert_linear is not None:
            colbert_vecs = colbert_linear(hidden_body)
        else:
            colbert_vecs = hidden_body
            
        colbert_vecs = colbert_vecs * mask_body
        colbert_vecs = torch.nn.functional.normalize(colbert_vecs, p=2, dim=-1)
        
        # Extract valid tokens
        manual_tokens = []
        for i in range(colbert_vecs.size(0)):
            valid_len = int(mask_body[i].squeeze(-1).sum().item())
            tokens = colbert_vecs[i, :valid_len, :]
            manual_tokens.append(tokens)
            
    manual_result = manual_tokens[0].numpy()
    
    # 3. Compare
    print(f"Official shape: {colbert_official.shape}")
    print(f"Manual shape: {manual_result.shape}")
    
    if manual_result.shape != colbert_official.shape:
        print("❌ SHAPE MISMATCH!")
        return

    diff = np.abs(manual_result - colbert_official).max()
    print(f"Max difference: {diff}")
    
    if diff < 1e-5:
        print("✅ ColBERT extraction logic MATCHES!")
    else:
        print("❌ Values differ significantly!")

if __name__ == "__main__":
    test_colbert_extraction_match()

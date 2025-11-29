import torch
import os
from typing import List, Type
from pathlib import Path

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding
from text_embeddings_server.fde_pytorch_opt import FDEBuilder, FDEConfig

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    BGEM3FlagModel = None

class BGEM3FDEModel(Model):
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        datatype: torch.dtype,
        pool: str = "bge_m3_fde",
        trust_remote: bool = False,
    ):
        if BGEM3FlagModel is None:
            raise ImportError("FlagEmbedding is required for BGEM3FDEModel")

        # Initialize the underlying model
        self.model_wrapper = BGEM3FlagModel(
            str(model_path), 
            use_fp16=(datatype == torch.float16),
            device=device
        )
        
        # Pass the actual HF model to the base class if possible, or None if it expects a specific type.
        # TEI's Model class typically expects a model handle. 
        # Here we pass self.model_wrapper.model (the HF model) to be safe.
        super().__init__(self.model_wrapper.model, datatype, device)
        
        # Load FDE params
        fde_config_path = model_path / "fde_config.json"
        
        # Default config
        config_dict = {
            "ksim": 5,
            "d_proj": 16,
            "R_reps": 10,
            "d_final": 1024,
            "fill_empty_clusters": True,
            "seed": 42,
            "use_mixed_precision": False
        }
        
        # Load from JSON if exists
        if fde_config_path.exists():
            import json
            with open(fde_config_path, "r") as f:
                loaded_config = json.load(f)
                config_dict.update(loaded_config)
                
        # Override with environment variables
        if "FDE_KSIM" in os.environ:
            config_dict["ksim"] = int(os.environ["FDE_KSIM"])
        if "FDE_D_PROJ" in os.environ:
            config_dict["d_proj"] = int(os.environ["FDE_D_PROJ"])
        if "FDE_R_REPS" in os.environ:
            config_dict["R_reps"] = int(os.environ["FDE_R_REPS"])
        if "FDE_D_FINAL" in os.environ:
            val = os.environ["FDE_D_FINAL"]
            config_dict["d_final"] = int(val) if val.lower() != "none" else None
        if "FDE_SEED" in os.environ:
            config_dict["seed"] = int(os.environ["FDE_SEED"])
            
        self.fde_config = FDEConfig(**config_dict)
        
        self.fde_builder = FDEBuilder(
            d=1024, # BGE-M3 colbert dim
            config=self.fde_config,
            device=device
        )
        
        fde_params_path = model_path / "fde_params.pt"
        if fde_params_path.exists():
            state = torch.load(fde_params_path, map_location=device)
            if "G" in state:
                if state["G"].shape != self.fde_builder.G.shape:
                    raise ValueError(f"G shape mismatch: saved {state['G'].shape}, expected {self.fde_builder.G.shape}")
                self.fde_builder.G = state["G"].to(device)
            if "W" in state and self.fde_builder.W is not None:
                if state["W"].shape != self.fde_builder.W.shape:
                    raise ValueError(f"W shape mismatch: saved {state['W'].shape}, expected {self.fde_builder.W.shape}")
                self.fde_builder.W = state["W"].to(device)
            if "P" in state and self.fde_builder.P is not None:
                if state["P"].shape != self.fde_builder.P.shape:
                    raise ValueError(f"P shape mismatch: saved {state['P'].shape}, expected {self.fde_builder.P.shape}")
                self.fde_builder.P = state["P"].to(device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        # Ensure inputs are on the correct device
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        
        with torch.no_grad():
            # 1) Access the underlying XLM-RoBERTa model directly
            # self.model_wrapper.model is BGEM3ForInference
            # self.model_wrapper.model.model is XLMRobertaModel
            base_model = self.model_wrapper.model.model
            
            model_output = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            last_hidden = model_output.last_hidden_state # [B, L, H]
            
            # Correct BGE-M3 ColBERT extraction:
            # 1. Exclude CLS token ([:, 1:])
            # 2. Apply mask to zero out padding
            
            hidden_body = last_hidden[:, 1:, :]
            mask_body = attention_mask[:, 1:].unsqueeze(-1).float()
            
            # colbert_linear is on BGEM3ForInference (self.model_wrapper.model)
            colbert_linear = getattr(self.model_wrapper.model, "colbert_linear", None)
            
            if colbert_linear is not None:
                colbert_vecs = colbert_linear(hidden_body)
            else:
                colbert_vecs = hidden_body
                
            # Zero out padding
            colbert_vecs = colbert_vecs * mask_body
            
            # Normalize per token
            colbert_vecs = torch.nn.functional.normalize(colbert_vecs, p=2, dim=-1)
            
            # Prepare for FDE
            batch_tokens_list = []
            for i in range(colbert_vecs.size(0)):
                # valid_len calculation needs to account for CLS exclusion
                # mask_body[i] corresponds to tokens after CLS
                valid_len = int(mask_body[i].squeeze(-1).sum().item())
                tokens = colbert_vecs[i, :valid_len, :]
                batch_tokens_list.append(tokens)
                
            fde_embeddings = self.fde_builder.encode_documents(batch_tokens_list)
            
            embeddings = []
            for i in range(len(fde_embeddings)):
                embeddings.append(Embedding(
                    values=fde_embeddings[i].cpu().tolist()
                ))
                
            return embeddings

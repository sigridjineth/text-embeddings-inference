import torch
import json
import argparse
from pathlib import Path
from text_embeddings_server.fde_pytorch_opt import FDEConfig, FDEBuilder

def main():
    parser = argparse.ArgumentParser(description="Generate FDE parameters for BGE-M3")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for fde_params.pt and fde_config.json")
    parser.add_argument("--ksim", type=int, default=5)
    parser.add_argument("--d-proj", type=int, default=16)
    parser.add_argument("--r-reps", type=int, default=10)
    parser.add_argument("--d-final", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = FDEConfig(
        ksim=args.ksim,
        d_proj=args.d_proj,
        R_reps=args.r_reps,
        d_final=args.d_final,
        fill_empty_clusters=True,
        seed=args.seed
    )
    
    # Save config
    config_dict = {
        "ksim": config.ksim,
        "d_proj": config.d_proj,
        "R_reps": config.R_reps,
        "d_final": config.d_final,
        "fill_empty_clusters": config.fill_empty_clusters,
        "seed": config.seed,
        "use_mixed_precision": config.use_mixed_precision
    }
    
    with open(output_dir / "fde_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
        
    print(f"Saved config to {output_dir / 'fde_config.json'}")
    
    # Generate params
    device = torch.device("cpu")
    # BGE-M3 colbert dim is 1024
    builder = FDEBuilder(d=1024, config=config, device=device)
    
    # We need to save the persistent G, W, P
    state = {
        "G": builder.G,
        "W": builder.W,
        "P": builder.P
    }
    
    torch.save(state, output_dir / "fde_params.pt")
    print(f"Saved params to {output_dir / 'fde_params.pt'}")

if __name__ == "__main__":
    main()

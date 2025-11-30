
import torch
from safetensors.torch import load_file
import sys

def check_params(filename="fde_params.safetensors"):
    print(f"Checking {filename} for NaNs/Infs...")
    try:
        tensors = load_file(filename)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return

    for k, v in tensors.items():
        print(f"Tensor '{k}': shape={v.shape}, dtype={v.dtype}")
        if torch.isnan(v).any():
            print(f"  [ERROR] {k} contains NaNs!")
        if torch.isinf(v).any():
            print(f"  [ERROR] {k} contains Infs!")
        
        # Check for zeros in W or P which might cause issues?
        if k == 'w' and (v == 0).all():
             print(f"  [WARN] {k} is all zeros!")
        if k == 'p' and (v == 0).all():
             print(f"  [WARN] {k} is all zeros!")

    print("Check complete.")

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "fde_params.safetensors"
    check_params(filename)

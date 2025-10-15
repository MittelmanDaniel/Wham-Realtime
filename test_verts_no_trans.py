"""Test what verts_cam looks like WITHOUT adding trans_cam"""
import joblib
import numpy as np
import torch
from configs.config import get_cfg_defaults
from lib.models import build_body_model, build_network

# Setup
cfg = get_cfg_defaults()
cfg.merge_from_file('configs/yamls/demo.yaml')

# Build models
smpl = build_body_model(cfg.DEVICE, 1)
network = build_network(cfg, smpl)
network.eval()

# Load one frame of results to get parameters
results = joblib.load('output/framebyframe_wham/smpl_results_final.pkl')
r0 = results[0]  # First frame

print("Testing verts computation...")
print(f"trans_cam from saved results: {r0['trans_cam']}")

# Recreate SMPL output
pose_tensor = torch.from_numpy(r0['pose']).float().to(cfg.DEVICE).unsqueeze(0).unsqueeze(0)
betas_tensor = torch.from_numpy(r0['betas']).float().to(cfg.DEVICE).unsqueeze(0).unsqueeze(0)

# Run through SMPL
output = smpl(pose_tensor, betas_tensor, cam=None)
print(f"\nDirect SMPL output:")
print(f"  vertices shape: {output.vertices.shape}")
print(f"  vertices range: [{output.vertices.min():.2f}, {output.vertices.max():.2f}]")
print(f"  vertices mean: {output.vertices[0].mean(dim=0).cpu().numpy()}")

# Compare to saved verts
print(f"\nSaved verts:")
print(f"  shape: {r0['verts'].shape}")
print(f"  range: [{r0['verts'].min():.2f}, {r0['verts'].max():.2f}]")
print(f"  mean: {r0['verts'].mean(axis=0)}")

print(f"\nAre they the same? {np.allclose(output.vertices[0].cpu().numpy(), r0['verts'], atol=0.01)}")


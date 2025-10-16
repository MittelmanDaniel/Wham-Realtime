"""Debug what's in the SMPL output"""
import joblib
import numpy as np

results = joblib.load('output/framebyframe_wham/smpl_results_final.pkl')
r0 = results[0]

print("Frame 0 SMPL output:")
print(f"  pose shape: {r0['pose'].shape}")
print(f"  betas shape: {r0['betas'].shape}")
print(f"  trans_cam: {r0['trans_cam']}")
print(f"  verts shape: {r0['verts'].shape}")
print(f"  verts range: [{r0['verts'].min():.2f}, {r0['verts'].max():.2f}]")
print(f"  verts mean: {r0['verts'].mean(axis=0)}")

# The verts should be (verts_cam + trans_cam)
# If trans_cam is [0.86, -1.07, 36.5] and verts range is [-1.15, 42.07],
# that means verts_cam alone was roughly [-2, 0, 5] before adding trans_cam

# Expected: verts should be in camera space, roughly centered around trans_cam
# with a span of ~1.7m (height of person) in the largest dimension

print("\n⚠️  PROBLEM ANALYSIS:")
print(f"  Person's Z depth: {r0['trans_cam'][2]:.1f} units")
print(f"  Verts Z range: [{r0['verts'][:, 2].min():.1f}, {r0['verts'][:, 2].max():.1f}]")
print(f"  Z span: {r0['verts'][:, 2].max() - r0['verts'][:, 2].min():.1f} units")
print()
print("Expected for proper rendering:")
print("  - Person Z depth should be 2-10 units for weak perspective")
print("  - Verts should span ~1.7 units in height")
print("  - But we have Z=41 and span=0.5!")
print()
print("This means the scale is COMPLETELY wrong!")



"""Check what SMPL pose parameters look like"""
import joblib
import numpy as np

results = joblib.load('output/framebyframe_wham/smpl_results_final.pkl')

print(f"Loaded {len(results)} results\n")

# Check first 5 frames
for i in range(min(5, len(results))):
    r = results[i]
    pose = r['pose']  # (144,) = 24 joints * 6D rotation
    
    print(f"Frame {r['frame_id']}:")
    print(f"  Pose shape: {pose.shape}")
    print(f"  Pose range: [{pose.min():.4f}, {pose.max():.4f}]")
    print(f"  Pose mean: {pose.mean():.4f}")
    print(f"  Pose std: {pose.std():.4f}")
    print(f"  Is all zeros? {np.allclose(pose, 0)}")
    print(f"  Is all ones? {np.allclose(pose, 1)}")
    
    # Check the first joint (global orientation)
    global_orient = pose[:6]
    print(f"  Global orient: {global_orient}")
    print()

# Check betas (shape parameters)
r0 = results[0]
print(f"\nShape parameters (betas):")
print(f"  Shape: {r0['betas'].shape}")
print(f"  Values: {r0['betas']}")
print(f"  All zeros? {np.allclose(r0['betas'], 0)}")



import joblib
import numpy as np

# Load results
results = joblib.load('output/framebyframe_wham/smpl_results_final.pkl')

print(f"Total results: {len(results)}")
print(f"\nFirst result:")
for key, value in results[0].items():
    if isinstance(value, np.ndarray):
        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
    else:
        print(f"  {key}: {value}")

print(f"\nChecking if 'verts' exists in results:")
print(f"  Has 'verts': {'verts' in results[0]}")

if 'verts' in results[0]:
    verts = results[0]['verts']
    print(f"\n Verts info:")
    print(f"  Shape: {verts.shape}")
    print(f"  Min: {verts.min()}, Max: {verts.max()}")
    print(f"  Mean: {verts.mean()}")
    print(f"  First 3 vertices:\n{verts[:3]}")


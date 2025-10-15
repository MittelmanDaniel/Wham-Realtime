"""Debug script to check vertex coordinates"""
import joblib
import numpy as np

# Load frame-by-frame results
print("Loading frame-by-frame results...")
fb_results = joblib.load('output/framebyframe_wham/smpl_results_final.pkl')
print(f"Loaded {len(fb_results)} frame-by-frame results")

# Check frame 100
fb_frame100 = [r for r in fb_results if r['frame_id'] == 100][0]
print("\nFrame-by-frame Frame 100:")
print(f"  verts shape: {fb_frame100['verts'].shape}")
print(f"  verts range: [{fb_frame100['verts'].min():.2f}, {fb_frame100['verts'].max():.2f}]")
print(f"  verts mean: {fb_frame100['verts'].mean(axis=0)}")
print(f"  trans_cam: {fb_frame100['trans_cam']}")

# Load demo results for comparison
print("\n" + "="*60)
print("Loading demo results...")
try:
    demo_results = joblib.load('output/demo_output.pkl')
    print(f"Loaded demo results with {len(demo_results)} subjects")
    
    # Get first subject
    first_subj = list(demo_results.keys())[0]
    demo_verts = demo_results[first_subj]['verts']
    print(f"\nDemo subject '{first_subj}':")
    print(f"  verts shape: {demo_verts.shape}")
    print(f"  verts range: [{demo_verts.min():.2f}, {demo_verts.max():.2f}]")
    print(f"  Frame 100 mean: {demo_verts[100].mean(axis=0)}")
    
    # Compare frame 100
    print("\n" + "="*60)
    print("COMPARISON for Frame 100:")
    print(f"Frame-by-frame verts mean: {fb_frame100['verts'].mean(axis=0)}")
    print(f"Demo verts mean:           {demo_verts[100].mean(axis=0)}")
    print(f"\nDifference: {fb_frame100['verts'].mean(axis=0) - demo_verts[100].mean(axis=0)}")
    
except FileNotFoundError:
    print("Demo results not found, skipping comparison")

# Check if vertices are reasonable for camera coordinates
print("\n" + "="*60)
print("Vertex coordinate analysis:")
print("For proper camera-space rendering, vertices should typically be:")
print("  X: around 0 (centered)")
print("  Y: around 0 (centered)")
print("  Z: positive, typically 100-500 cm from camera")
print(f"\nActual frame-by-frame verts:")
print(f"  X range: [{fb_frame100['verts'][:, 0].min():.2f}, {fb_frame100['verts'][:, 0].max():.2f}]")
print(f"  Y range: [{fb_frame100['verts'][:, 1].min():.2f}, {fb_frame100['verts'][:, 1].max():.2f}]")
print(f"  Z range: [{fb_frame100['verts'][:, 2].min():.2f}, {fb_frame100['verts'][:, 2].max():.2f}]")


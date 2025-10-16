"""Compare trans_cam from demo vs frame-by-frame"""
import joblib
import numpy as np

# Load frame-by-frame results
fb_results = joblib.load('output/framebyframe_wham/smpl_results_final.pkl')

print("Frame-by-Frame trans_cam (first 5 frames):")
for i in range(min(5, len(fb_results))):
    r = fb_results[i]
    print(f"Frame {r['frame_id']}: trans_cam = {r['trans_cam']}")

print("\nAnalysis:")
fb_trans = np.array([r['trans_cam'] for r in fb_results[:20]])
print(f"  X range: [{fb_trans[:, 0].min():.2f}, {fb_trans[:, 0].max():.2f}]")
print(f"  Y range: [{fb_trans[:, 1].min():.2f}, {fb_trans[:, 1].max():.2f}]")
print(f"  Z range: [{fb_trans[:, 2].min():.2f}, {fb_trans[:, 2].max():.2f}]")

print("\nExpected for weak perspective camera:")
print("  X, Y: should be small (within ±5 for centered person)")
print("  Z: should be positive, typically 2-10 for normalized coordinates")
print("\n⚠️  Our values are WAY too large! Z=36, X=-16, Y=-9")
print("⚠️  This suggests trans_cam is in PIXEL space, not normalized camera space!")



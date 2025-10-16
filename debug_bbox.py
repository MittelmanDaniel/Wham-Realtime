"""Debug bbox computation"""
import joblib

# Load frame 0 smpl results to see what we computed
results = joblib.load('output/framebyframe_wham/smpl_results_final.pkl')
r0 = results[0]

print("Frame 0 data:")
print(f"  trans_cam: {r0['trans_cam']}")

# Load timing stats to see frame info
timing = joblib.load('output/framebyframe_wham/timing_stats.pkl')
print(f"\nVideo info:")
print(f"  Width: {timing.get('width', 'N/A')}")
print(f"  Height: {timing.get('height', 'N/A')}")

# The bbox scale should be: max(bbox_width, bbox_height) / max(img_width, img_height)
# If person takes up 50% of frame, scale ≈ 0.5
# If person takes up full frame, scale ≈ 1.0

print("\nExpected behavior:")
print("  bbox_scale should be 0.3-0.8 for a typical person in frame")
print("  bbox_scale * 200 = 60-160 pixels (used as bbox_height in trans_cam calculation)")
print("  But our trans_cam suggests bbox_height is being computed wrong!")



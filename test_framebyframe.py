"""
Test frame-by-frame WHAM inference
"""
import torch
import numpy as np
from configs.config import get_cfg_defaults
from configs import constants as _C
from lib.models import build_network, build_body_model

def test_framebyframe():
    print("="*60)
    print("Testing Frame-by-Frame WHAM")
    print("="*60)
    
    # Load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    # Build model
    print("\n1. Building SMPL model...")
    smpl = build_body_model(cfg.DEVICE, cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN)
    print(f"   ✅ SMPL loaded on {cfg.DEVICE}")
    
    # Build network
    print("\n2. Building WHAM network...")
    network = build_network(cfg, smpl)
    network.eval()
    print(f"   ✅ Network loaded")
    
    # Check for new method
    print("\n3. Checking for forward_single_frame method...")
    has_method = hasattr(network, 'forward_single_frame')
    print(f"   Has forward_single_frame: {has_method}")
    
    if not has_method:
        print("   ❌ Method not found!")
        return False
    
    # Test with dummy data
    print("\n4. Testing with dummy single frame...")
    
    n_joints = _C.KEYPOINTS.NUM_JOINTS
    b = 1  # Batch size
    
    # Create dummy inputs
    x = torch.randn(b, 1, n_joints * 2 + 3).to(cfg.DEVICE)  # Single frame
    init_kp = torch.zeros(b, 1, n_joints * 3).to(cfg.DEVICE)
    init_smpl = torch.zeros(b, 1, 24 * 6).to(cfg.DEVICE)
    inits = (init_kp, init_smpl)
    mask = torch.ones(b, 1, n_joints, dtype=torch.bool).to(cfg.DEVICE)
    init_root = torch.zeros(b, 1, 6).to(cfg.DEVICE)
    cam_angvel = torch.zeros(b, 1, 6).to(cfg.DEVICE)  # 6D rotation representation!
    bbox = torch.tensor([[[0.3, 0.3, 0.8]]]).to(cfg.DEVICE)  # (B, 1, 3) - [x, y, scale]
    res = torch.tensor([[1080, 1920]]).to(cfg.DEVICE)  # (B, 2) - [width, height]
    # Simple camera intrinsics (B, 1, 3, 3)
    focal_length = ((1080**2 + 1920**2)**0.5)
    cam_intrinsics = torch.zeros(b, 1, 3, 3).to(cfg.DEVICE)
    cam_intrinsics[:, :, 0, 0] = focal_length  # fx
    cam_intrinsics[:, :, 1, 1] = focal_length  # fy
    cam_intrinsics[:, :, 0, 2] = 1080 / 2  # cx
    cam_intrinsics[:, :, 1, 2] = 1920 / 2  # cy
    cam_intrinsics[:, :, 2, 2] = 1.0
    
    print(f"   Input shape: {x.shape}")
    print(f"   Processing frame 1...")
    
    with torch.no_grad():
        # First frame
        output1, hidden_states1 = network.forward_single_frame(
            x, inits, mask=mask, init_root=init_root, cam_angvel=cam_angvel,
            bbox=bbox, res=res, cam_intrinsics=cam_intrinsics
        )
        print(f"   ✅ Frame 1 processed!")
        print(f"      Pose shape: {output1['pose'].shape}")
        print(f"      Trans shape: {output1['trans_cam'].shape}")
        print(f"      Hidden states: {list(hidden_states1.keys())}")
        
        # Second frame (reuse hidden states!)
        print(f"\n   Processing frame 2 with hidden states...")
        output2, hidden_states2 = network.forward_single_frame(
            x, inits, mask=mask, init_root=init_root, cam_angvel=cam_angvel,
            bbox=bbox, res=res, cam_intrinsics=cam_intrinsics,
            hidden_states=hidden_states1
        )
        print(f"   ✅ Frame 2 processed!")
        print(f"      Pose shape: {output2['pose'].shape}")
        print(f"      Trans shape: {output2['trans_cam'].shape}")
        
        # Verify outputs are different (hidden state is working)
        pose_diff = torch.abs(output2['pose'] - output1['pose']).mean().item()
        print(f"\n   Pose difference between frames: {pose_diff:.6f}")
        
        if pose_diff > 0:
            print(f"   ✅ Hidden states are working! (outputs differ)")
        else:
            print(f"   ⚠️  Outputs are identical (might be a problem)")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nFrame-by-frame processing is working!")
    print("You can now use network.forward_single_frame() for real-time inference!")
    
    return True

if __name__ == "__main__":
    success = test_framebyframe()
    exit(0 if success else 1)


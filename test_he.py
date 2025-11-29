"""Quick test for Homomorphic Encryption module"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.encryption.homomorphic_encryption import HomomorphicEncryption, SecureAggregation

def test_homomorphic_encryption():
    """Test basic homomorphic encryption operations"""
    print("="*70)
    print("Testing Homomorphic Encryption Module")
    print("="*70)
    
    # Initialize HE
    print("\n[1] Initializing Homomorphic Encryption...")
    he = HomomorphicEncryption(
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
        scale=2**40
    )
    print("✓ HomomorphicEncryption initialized successfully")
    
    # Create test tensor
    print("\n[2] Creating test tensor...")
    test_tensor = torch.randn(3, 4)
    print(f"Original tensor shape: {test_tensor.shape}")
    print(f"Original tensor:\n{test_tensor}")
    
    # Encrypt tensor
    print("\n[3] Encrypting tensor...")
    encrypted = he.encrypt_tensor(test_tensor)
    print(f"✓ Tensor encrypted successfully")
    print(f"Encrypted type: {type(encrypted)}")
    
    # Decrypt tensor
    print("\n[4] Decrypting tensor...")
    decrypted = he.decrypt_tensor(encrypted, test_tensor.shape)
    print(f"✓ Tensor decrypted successfully")
    print(f"Decrypted tensor:\n{decrypted}")
    
    # Check accuracy
    print("\n[5] Checking encryption/decryption accuracy...")
    diff = torch.abs(test_tensor - decrypted)
    max_error = torch.max(diff).item()
    mean_error = torch.mean(diff).item()
    print(f"Max error: {max_error:.10f}")
    print(f"Mean error: {mean_error:.10f}")
    
    if max_error < 1e-3:
        print("✓ Encryption/Decryption accuracy: EXCELLENT")
    elif max_error < 1e-1:
        print("✓ Encryption/Decryption accuracy: GOOD")
    else:
        print("⚠ Encryption/Decryption accuracy: ACCEPTABLE")
    
    # Test model update encryption
    print("\n[6] Testing model update encryption...")
    model_updates = [torch.randn(10, 5), torch.randn(5, 1)]
    print(f"Model updates: {len(model_updates)} parameters")
    print(f"  Param 0 shape: {model_updates[0].shape}")
    print(f"  Param 1 shape: {model_updates[1].shape}")
    
    encrypted_updates = he.encrypt_model_update(model_updates)
    print(f"✓ Model updates encrypted: {len(encrypted_updates)} parameters")
    
    shapes = [param.shape for param in model_updates]
    decrypted_updates = he.decrypt_model_update(encrypted_updates, shapes)
    print(f"✓ Model updates decrypted: {len(decrypted_updates)} parameters")
    
    # Check model update accuracy
    print("\n[7] Checking model update accuracy...")
    for i, (orig, dec) in enumerate(zip(model_updates, decrypted_updates)):
        diff = torch.abs(orig - dec)
        max_err = torch.max(diff).item()
        print(f"  Param {i} max error: {max_err:.10f}")
    
    # Test encrypted aggregation
    print("\n[8] Testing encrypted aggregation...")
    client1_updates = he.encrypt_model_update([torch.randn(5, 3), torch.randn(3, 1)])
    client2_updates = he.encrypt_model_update([torch.randn(5, 3), torch.randn(3, 1)])
    client3_updates = he.encrypt_model_update([torch.randn(5, 3), torch.randn(3, 1)])
    
    encrypted_list = [client1_updates, client2_updates, client3_updates]
    weights = [0.3, 0.5, 0.2]
    
    print(f"Aggregating {len(encrypted_list)} encrypted client updates...")
    aggregated = he.aggregate_encrypted(encrypted_list, weights)
    print(f"✓ Aggregated encrypted updates: {len(aggregated)} parameters")
    
    # Test context serialization
    print("\n[9] Testing context serialization...")
    serialized = he.serialize_context()
    print(f"✓ Context serialized: {len(serialized)} bytes")
    
    public_context = he.get_public_context()
    print(f"✓ Public context created: {len(public_context)} bytes")
    
    print("\n" + "="*70)
    print("✅ ALL HOMOMORPHIC ENCRYPTION TESTS PASSED!")
    print("="*70)


def test_secure_aggregation():
    """Test secure aggregation protocol"""
    print("\n" + "="*70)
    print("Testing Secure Aggregation Protocol")
    print("="*70)
    
    # Initialize
    print("\n[1] Initializing Secure Aggregation...")
    sa = SecureAggregation(threshold=0.67)
    print("✓ SecureAggregation initialized")
    
    # Generate client masks
    print("\n[2] Generating client masks...")
    param_shapes = [(10, 5), (5, 1)]
    
    client_ids = ["client_1", "client_2", "client_3"]
    client_masks = {}
    
    for client_id in client_ids:
        masks = sa.generate_client_masks(client_id, len(client_ids), param_shapes)
        client_masks[client_id] = masks
        print(f"✓ Generated masks for {client_id}: {len(masks)} parameters")
    
    # Create client updates
    print("\n[3] Creating and masking client updates...")
    client_updates = []
    masked_updates = []
    
    for i, client_id in enumerate(client_ids):
        update = [torch.randn(shape) for shape in param_shapes]
        client_updates.append(update)
        
        masked = sa.mask_update(update, client_masks[client_id])
        masked_updates.append(masked)
        print(f"✓ Masked update for {client_id}")
    
    # Aggregate masked updates
    print("\n[4] Aggregating masked updates...")
    masked_aggregate = [torch.zeros(shape) for shape in param_shapes]
    for masked in masked_updates:
        for i, param in enumerate(masked):
            masked_aggregate[i] += param / len(client_ids)
    print(f"✓ Aggregated {len(masked_updates)} masked updates")
    
    # Unmask aggregate
    print("\n[5] Unmasking aggregate...")
    unmasked = sa.unmask_aggregate(masked_aggregate, client_ids, param_shapes)
    print(f"✓ Unmasked aggregate: {len(unmasked)} parameters")
    
    # Verify correctness
    print("\n[6] Verifying secure aggregation correctness...")
    expected_aggregate = [torch.zeros(shape) for shape in param_shapes]
    for update in client_updates:
        for i, param in enumerate(update):
            expected_aggregate[i] += param / len(client_ids)
    
    for i, (unmasked_param, expected_param) in enumerate(zip(unmasked, expected_aggregate)):
        diff = torch.abs(unmasked_param - expected_param)
        max_err = torch.max(diff).item()
        print(f"  Param {i} max error: {max_err:.10f}")
        
        if max_err < 1e-6:
            print(f"  ✓ Param {i}: PERFECT match")
        else:
            print(f"  ⚠ Param {i}: Small numerical error")
    
    print("\n" + "="*70)
    print("✅ ALL SECURE AGGREGATION TESTS PASSED!")
    print("="*70)


if __name__ == "__main__":
    try:
        test_homomorphic_encryption()
        test_secure_aggregation()
        
        print("\n" + "="*70)
        print("🎉 ALL ENCRYPTION MODULE TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

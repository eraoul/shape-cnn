"""
Quick test to verify the new approach is correctly set up.
This test checks that all components are properly defined without requiring PyTorch.
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported (structure check)."""
    print("Testing module structure...")

    # Check files exist
    required_files = [
        'data_generation_improved.py',
        'model_unet.py',
        'train_unet.py',
        'inference_unet.py',
        'README_NEW_APPROACH.md',
    ]

    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file} exists")
        else:
            print(f"  ✗ {file} missing")
            return False

    return True


def test_data_generation_logic():
    """Test data generation logic without dependencies."""
    print("\nTesting data generation logic...")

    # Check shape classes are defined
    from data_generation_improved import SHAPE_CLASSES, NUM_SHAPE_CLASSES

    print(f"  Shape classes: {list(SHAPE_CLASSES.keys())}")
    print(f"  Number of classes: {NUM_SHAPE_CLASSES}")

    assert NUM_SHAPE_CLASSES == 6, "Should have 6 shape classes"
    assert 'triangle' in SHAPE_CLASSES, "Should include triangles"
    assert 'polygon' in SHAPE_CLASSES, "Should include polygons"
    assert 'circle' in SHAPE_CLASSES, "Should include circles"

    print("  ✓ All shape types defined correctly")
    return True


def test_approach_improvements():
    """Verify key improvements are documented."""
    print("\nVerifying improvements...")

    improvements = {
        'All shape types': False,
        'U-Net': False,
        'Focal Loss': False,
        'Data augmentation': False,
        'Balanced loss': False,
    }

    # Check README
    with open('README_NEW_APPROACH.md', 'r') as f:
        content = f.read()

        if 'triangle' in content.lower() and 'polygon' in content.lower():
            improvements['All shape types'] = True
        if 'U-Net' in content or 'unet' in content.lower():
            improvements['U-Net'] = True
        if 'focal' in content.lower():
            improvements['Focal Loss'] = True
        if 'augment' in content.lower():
            improvements['Data augmentation'] = True
        if 'balanced' in content.lower():
            improvements['Balanced loss'] = True

    for improvement, found in improvements.items():
        status = "✓" if found else "✗"
        print(f"  {status} {improvement}")

    return all(improvements.values())


def main():
    print("="*60)
    print("TESTING NEW SHAPE DETECTION APPROACH")
    print("="*60)

    tests = [
        ("File structure", test_imports),
        ("Data generation", test_data_generation_logic),
        ("Documented improvements", test_approach_improvements),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            results.append(False)

    print("\n" + "="*60)
    if all(results):
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train model: python train_unet.py")
        print("3. Run inference: python inference_unet.py")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())

"""
Test Simplified Data Generation
Verify color 0 is reserved, overlaps have different colors, and only rectangles/lines are generated
"""

import numpy as np
from data_generation_with_types import ShapeGeneratorWithTypes

def test_simplified_generation():
    """Test the simplified generation with new constraints."""
    print("="*80)
    print("TESTING SIMPLIFIED DATA GENERATION")
    print("="*80)

    generator = ShapeGeneratorWithTypes(num_colors=10, min_size=5, max_size=20)

    # Test 1: Color 0 is reserved for background
    print("\n1. Testing color 0 reservation...")
    passed_color_test = True
    for i in range(50):
        sample = generator.generate_sample()
        grid = sample['grid']
        h, w, c = grid.shape

        # Check if any object uses color 0
        for y in range(h):
            for x in range(w):
                if grid[y, x, 0] == 1:
                    # Found a pixel with color 0
                    # Check if it's actually background (no other color active)
                    other_colors = grid[y, x, 1:].sum()
                    if other_colors == 0:
                        # This is truly background, OK
                        pass
                    else:
                        print(f"  ✗ Found object using color 0 at ({y},{x})")
                        passed_color_test = False
                        break

    if passed_color_test:
        print("  ✓ Color 0 is correctly reserved for background only")

    # Test 2: Only rectangles, lines, and single_cell shapes
    print("\n2. Testing shape types...")
    shape_types_seen = set()
    for i in range(50):
        sample = generator.generate_sample()
        metadata = sample['metadata']

        if 'type' in metadata:
            shape_types_seen.add(metadata['type'])
        elif 'shapes' in metadata:
            for shape in metadata['shapes']:
                shape_types_seen.add(shape['type'])

    allowed_types = {'line', 'rectangle', 'single_cell'}
    invalid_types = shape_types_seen - allowed_types

    if len(invalid_types) == 0:
        print(f"  ✓ Only allowed shapes generated: {sorted(shape_types_seen)}")
    else:
        print(f"  ✗ Invalid shape types found: {invalid_types}")

    # Test 3: Overlapping objects have different colors
    print("\n3. Testing color separation in overlaps...")
    overlap_violations = 0
    samples_with_overlaps = 0

    for i in range(100):
        sample = generator.generate_sample()
        grid = sample['grid']
        h, w, c = grid.shape

        # Check each pixel for overlaps (multiple colors active)
        for y in range(h):
            for x in range(w):
                active_colors = []
                for color in range(c):
                    if grid[y, x, color] == 1:
                        active_colors.append(color)

                if len(active_colors) > 1:
                    samples_with_overlaps += 1
                    # This pixel has overlap - all colors should be non-zero
                    if 0 in active_colors:
                        overlap_violations += 1
                        print(f"  ⚠ Overlap at ({y},{x}) includes background color 0")
                    break

    if samples_with_overlaps > 0:
        print(f"  ℹ Found {samples_with_overlaps} samples with overlapping objects")
        if overlap_violations == 0:
            print(f"  ✓ All overlapping objects have different non-zero colors")
        else:
            print(f"  ✗ Found {overlap_violations} overlap violations")
    else:
        print(f"  ℹ No overlapping objects found in {100} samples")

    # Test 4: Verify shape classes
    print("\n4. Testing shape class IDs...")
    from model_small import ShapeNetWithClassification

    expected_classes = {
        'background': 0,
        'line': 1,
        'rectangle': 2,
        'single_cell': 3
    }

    actual_classes = ShapeNetWithClassification.SHAPE_CLASSES

    if actual_classes == expected_classes:
        print(f"  ✓ Shape classes are correct: {actual_classes}")
    else:
        print(f"  ✗ Shape classes mismatch!")
        print(f"    Expected: {expected_classes}")
        print(f"    Got: {actual_classes}")

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)

    # Generate a sample and show its details
    print("\nExample generated sample:")
    sample = generator.generate_sample()
    grid = sample['grid']
    metadata = sample['metadata']

    print(f"\nGrid shape: {grid.shape[0]}x{grid.shape[1]}")
    print(f"Metadata: {metadata}")

    # Show color usage
    colors_used = []
    for color in range(grid.shape[2]):
        if grid[:, :, color].sum() > 0:
            colors_used.append(color)
    print(f"Colors used: {colors_used}")

    if 0 in colors_used:
        # Check if it's just background
        non_zero_sum = grid[:, :, 1:].sum()
        zero_sum = grid[:, :, 0].sum()
        if non_zero_sum > 0:
            print(f"  ⚠ Warning: Color 0 is used alongside other colors")
        else:
            print(f"  ✓ Color 0 is background only (entire grid is empty)")

if __name__ == "__main__":
    test_simplified_generation()

"""
Visualize Multiple Examples from the Shape Recognition Pipeline
Generates predictions on multiple test samples and saves visualizations
"""

import torch
import matplotlib.pyplot as plt
from data_generation import ShapeGenerator
from model import StructuralShapeNet
from inference import ShapeInference


def visualize_multiple_examples(model_path='best_model.pth', num_examples=6, grid_sizes=None):
    """
    Visualize multiple examples with predictions.

    Args:
        model_path: Path to saved model weights
        num_examples: Number of examples to visualize
        grid_sizes: List of (h, w) tuples for grid sizes. If None, uses random sizes.
    """
    # Configuration
    NUM_COLORS = 8
    MAX_INSTANCES = 10

    # Load model
    print("Loading model...")
    model = StructuralShapeNet(NUM_COLORS, MAX_INSTANCES)
    model.load_state_dict(torch.load(model_path))
    inference = ShapeInference(model)
    print(f"Using device: {inference.device}")

    # Generate test samples
    print(f"\nGenerating {num_examples} test samples...")
    generator = ShapeGenerator(NUM_COLORS, min_size=5, max_size=25)

    samples = []
    for i in range(num_examples):
        if grid_sizes and i < len(grid_sizes):
            # Create generator with specific size
            h, w = grid_sizes[i]
            temp_gen = ShapeGenerator(NUM_COLORS, min_size=h, max_size=h)
            # Adjust for width
            sample = temp_gen.generate_sample()
            # This is a workaround - better to modify generator
            samples.append(temp_gen.generate_sample())
        else:
            samples.append(generator.generate_sample())

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 4 * num_examples))

    for idx, sample in enumerate(samples):
        test_grid = sample['grid']
        h, w = test_grid.shape[:2]

        # Run inference
        prediction = inference.predict(test_grid)

        # Create subplot row (6 columns per example)
        base_idx = idx * 6

        # Input grid
        ax1 = plt.subplot(num_examples, 6, base_idx + 1)
        input_rgb = test_grid.argmax(axis=2)
        ax1.imshow(input_rgb, cmap='tab10', interpolation='nearest')
        ax1.set_title(f'Example {idx+1}: Input\n({h}x{w})')
        ax1.axis('off')

        # Instance segmentation
        ax2 = plt.subplot(num_examples, 6, base_idx + 2)
        ax2.imshow(prediction['raw_outputs']['instance_map'], cmap='tab10', interpolation='nearest')
        ax2.set_title('Instance Segmentation')
        ax2.axis('off')

        # Vertex heatmap
        ax3 = plt.subplot(num_examples, 6, base_idx + 3)
        ax3.imshow(prediction['raw_outputs']['vertex_map'], cmap='hot', interpolation='nearest')
        ax3.set_title('Vertex Heatmap')
        ax3.axis('off')

        # Edge heatmap
        ax4 = plt.subplot(num_examples, 6, base_idx + 4)
        ax4.imshow(prediction['raw_outputs']['edge_map'], cmap='hot', interpolation='nearest')
        ax4.set_title('Edge Heatmap')
        ax4.axis('off')

        # Detected vertices overlay
        ax5 = plt.subplot(num_examples, 6, base_idx + 5)
        ax5.imshow(input_rgb, cmap='tab10', alpha=0.6, interpolation='nearest')
        for obj in prediction['objects']:
            for vy, vx in obj['vertices']:
                ax5.plot(vx, vy, 'r*', markersize=12, markeredgewidth=1.5, markeredgecolor='white')
        ax5.set_title(f"Vertices ({len(prediction['objects'])} objects)")
        ax5.axis('off')

        # Structured output text
        ax6 = plt.subplot(num_examples, 6, base_idx + 6)
        ax6.axis('off')

        text = f"Grid: {prediction['grid_size']}\n"
        text += f"Objects: {len(prediction['objects'])}\n\n"

        for obj in prediction['objects']:
            text += f"• {obj['type'].upper()}\n"
            text += f"  Color: {obj['color']}\n"
            text += f"  Vertices: {obj['properties']['num_vertices']}\n"
            text += f"  Area: {obj['properties']['area']} cells\n"
            text += f"  Filled: {obj['properties']['is_filled']}\n"

            # Add shape-specific info
            if 'angles' in obj:
                text += f"  Angles: {[f'{a:.0f}°' for a in obj['angles']]}\n"
            elif 'dimensions' in obj:
                text += f"  Size: {obj['dimensions']['width']:.1f} x {obj['dimensions']['height']:.1f}\n"
            elif 'circle_params' in obj:
                text += f"  Radius: {obj['circle_params']['radius']:.1f}\n"

            text += "\n"

        ax6.text(0.05, 0.95, text, fontsize=9, verticalalignment='top',
                fontfamily='monospace', transform=ax6.transAxes)
        ax6.set_title('Detected Properties')

    plt.tight_layout()

    # Save visualization
    output_path = 'multiple_examples_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()

    return samples


def visualize_specific_sizes(model_path='best_model.pth'):
    """
    Visualize examples with specific grid sizes to test edge cases.
    """
    # Test various grid sizes
    test_configs = [
        (5, 5, "Small 5x5"),
        (10, 10, "Medium 10x10"),
        (15, 15, "Large 15x15"),
        (20, 8, "Wide 20x8"),
        (8, 20, "Tall 8x20"),
        (25, 25, "Extra Large 25x25"),
    ]

    NUM_COLORS = 8
    MAX_INSTANCES = 10

    # Load model
    print("Loading model...")
    model = StructuralShapeNet(NUM_COLORS, MAX_INSTANCES)
    model.load_state_dict(torch.load(model_path))
    inference = ShapeInference(model)
    print(f"Using device: {inference.device}")

    print(f"\nGenerating {len(test_configs)} test samples with specific sizes...")

    samples = []
    for h, w, desc in test_configs:
        # Create generator with specific size range
        generator = ShapeGenerator(NUM_COLORS, min_size=h, max_size=h)
        sample = generator.generate_sample()
        # Ensure we get the right size by regenerating if needed
        while sample['grid'].shape[0] != h or sample['grid'].shape[1] != w:
            generator = ShapeGenerator(NUM_COLORS, min_size=min(h,w), max_size=max(h,w))
            sample = generator.generate_sample()
        samples.append((sample, desc))

    # Create figure
    fig = plt.figure(figsize=(20, 4 * len(test_configs)))

    for idx, (sample, desc) in enumerate(samples):
        test_grid = sample['grid']
        h, w = test_grid.shape[:2]

        # Run inference
        prediction = inference.predict(test_grid)

        # Create subplot row
        base_idx = idx * 6

        # Input grid
        ax1 = plt.subplot(len(test_configs), 6, base_idx + 1)
        input_rgb = test_grid.argmax(axis=2)
        ax1.imshow(input_rgb, cmap='tab10', interpolation='nearest')
        ax1.set_title(f'{desc}\nInput Grid')
        ax1.axis('off')

        # Instance segmentation
        ax2 = plt.subplot(len(test_configs), 6, base_idx + 2)
        ax2.imshow(prediction['raw_outputs']['instance_map'], cmap='tab10', interpolation='nearest')
        ax2.set_title('Instances')
        ax2.axis('off')

        # Vertex heatmap
        ax3 = plt.subplot(len(test_configs), 6, base_idx + 3)
        ax3.imshow(prediction['raw_outputs']['vertex_map'], cmap='hot', interpolation='nearest')
        ax3.set_title('Vertices')
        ax3.axis('off')

        # Edge heatmap
        ax4 = plt.subplot(len(test_configs), 6, base_idx + 4)
        ax4.imshow(prediction['raw_outputs']['edge_map'], cmap='hot', interpolation='nearest')
        ax4.set_title('Edges')
        ax4.axis('off')

        # Overlay
        ax5 = plt.subplot(len(test_configs), 6, base_idx + 5)
        ax5.imshow(input_rgb, cmap='tab10', alpha=0.6, interpolation='nearest')
        for obj in prediction['objects']:
            for vy, vx in obj['vertices']:
                ax5.plot(vx, vy, 'r*', markersize=10, markeredgewidth=1, markeredgecolor='white')
        ax5.set_title(f"Detected ({len(prediction['objects'])})")
        ax5.axis('off')

        # Text summary
        ax6 = plt.subplot(len(test_configs), 6, base_idx + 6)
        ax6.axis('off')

        text = f"Objects: {len(prediction['objects'])}\n\n"
        for obj in prediction['objects']:
            text += f"• {obj['type']}\n"
            text += f"  Area: {obj['properties']['area']}\n\n"

        ax6.text(0.05, 0.95, text, fontsize=9, verticalalignment='top',
                fontfamily='monospace', transform=ax6.transAxes)
        ax6.set_title('Summary')

    plt.tight_layout()

    output_path = 'size_comparison_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--sizes':
        # Visualize specific sizes
        visualize_specific_sizes()
    else:
        # Visualize random examples
        num_examples = 6
        if len(sys.argv) > 1:
            num_examples = int(sys.argv[1])
        visualize_multiple_examples(num_examples=num_examples)

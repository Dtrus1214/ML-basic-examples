"""
Convolution Layer - Raw Implementation (No PyTorch/scipy conv)
==============================================================
Educational demo showing how a 2D convolution works step-by-step.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. CREATE SAMPLE IMAGES (simple 8x8 patterns)
# ---------------------------------------------------------------------------
def sample_vertical_edge():
    """Left=0, right=1. ONE vertical boundary. Vertical edge kernel responds here."""
    return np.array([
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ], dtype=np.float32)


def sample_horizontal_edge():
    """Top=0, bottom=1. ONE horizontal boundary. Horizontal edge kernel responds here."""
    return np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.float32)


def sample_blur_test():
    """Checkerboard for blur demo - many sharp transitions get smoothed."""
    return np.array([
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# 2. CONVOLUTION - RAW IMPLEMENTATION (manual sliding window)
# ---------------------------------------------------------------------------
def conv2d_raw(image, kernel, stride=1):
    """
    Convolve image with kernel using raw Python logic.
    
    At each position:
      1. Extract a patch of image same size as kernel
      2. Multiply patch element-wise with kernel
      3. Sum all products -> one output value
    
    Args:
        image: 2D array (H x W)
        kernel: 2D filter (Kh x Kw)
        stride: step size when sliding
    """
    H, W = image.shape
    Kh, Kw = kernel.shape
    
    # Output size formula: (H - Kh) / stride + 1
    out_h = (H - Kh) // stride + 1
    out_w = (W - Kw) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            # Top-left corner of current patch
            y_start = i * stride
            x_start = j * stride
            
            # Extract patch
            patch = image[y_start : y_start + Kh, x_start : x_start + Kw]
            
            # Element-wise multiply and sum (this IS the convolution at this position)
            output[i, j] = np.sum(patch * kernel)
    
    return output


def conv2d_raw_with_padding(image, kernel, stride=1, padding=0):
    """
    Same as above but with padding support (like real Conv2d).
    Padding adds zeros around the image.
    """
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    return conv2d_raw(image, kernel, stride)


# ---------------------------------------------------------------------------
# 3. EXAMPLE KERNELS (filters)
# ---------------------------------------------------------------------------

# Vertical edge detector: responds to horizontal changes (left-right)
KERNEL_VERTICAL_EDGE = np.array([
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1],
], dtype=np.float32)

# Horizontal edge detector: responds to vertical changes (up-down)
KERNEL_HORIZONTAL_EDGE = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1],
], dtype=np.float32)

# Blur / smoothing filter
KERNEL_BLUR = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
], dtype=np.float32)

# Sharpen filter
KERNEL_SHARPEN = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0],
], dtype=np.float32)


# ---------------------------------------------------------------------------
# 4. MAIN - DEMO
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("CONVOLUTION LAYER - Raw Implementation Demo")
    print("=" * 60)
    
    # Vertical edge image: left=0, right=1 (one vertical boundary)
    img_vert = sample_vertical_edge()
    print("\n[1] Sample: Vertical edge (left=0, right=1):")
    print(img_vert)
    
    # Horizontal edge image: top=0, bottom=1 (one horizontal boundary)
    img_horz = sample_horizontal_edge()
    print("\n    Sample: Horizontal edge (top=0, bottom=1):")
    print(img_horz)
    
    # Show one convolution step manually (use vertical image + vertical kernel)
    print("\n[2] How ONE output value is computed (vertical kernel on vertical-edge image):")
    kernel = KERNEL_VERTICAL_EDGE
    patch = img_vert[0:3, 2:5]  # patch spanning the vertical boundary
    result = np.sum(patch * kernel)
    print(f"    Kernel (vertical edge detector):")
    print(kernel)
    print(f"    Patch at position (0,2) - spans the vertical boundary:")
    print(patch)
    print(f"    Element-wise multiply & sum: {result}")
    
    # Full convolution
    print("\n[3] Vertical edge kernel on vertical-edge image (strong response at boundary):")
    out = conv2d_raw(img_vert, kernel)
    print(out)
    print(f"    Output shape: {out.shape} (smaller than input due to no padding)")
    
    # With padding to preserve size
    print("\n[4] Same conv with padding=1 (preserves spatial size):")
    out_padded = conv2d_raw_with_padding(img_vert, kernel, padding=1)
    print(out_padded)
    print(f"    Output shape: {out_padded.shape}")
    
    # Each kernel on its matching image
    print("\n[5] Each kernel on its matching image (clear separation):")
    print("    Vertical kernel on vertical-edge image (left|right boundary):")
    print(conv2d_raw(img_vert, KERNEL_VERTICAL_EDGE))
    print("    Horizontal kernel on horizontal-edge image (top|bottom boundary):")
    print(conv2d_raw(img_horz, KERNEL_HORIZONTAL_EDGE))
    print("    Blur kernel on checkerboard (smooths sharp transitions):")
    print(conv2d_raw(sample_blur_test(), KERNEL_BLUR))
    
    # Optional: visualize with matplotlib if available
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(10, 7))
        
        # Row 1: Vertical edge kernel on vertical-edge image
        axes[0, 0].imshow(img_vert, cmap='gray')
        axes[0, 0].set_title('Input: Vertical edge (left|right)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(KERNEL_VERTICAL_EDGE, cmap='RdBu')
        axes[0, 1].set_title('Kernel: Vertical Edge')
        axes[0, 1].axis('off')
        
        out_v = conv2d_raw(img_vert, KERNEL_VERTICAL_EDGE)
        axes[0, 2].imshow(out_v, cmap='gray')
        axes[0, 2].set_title('Output: responds at boundary')
        axes[0, 2].axis('off')
        
        # Row 2: Horizontal edge kernel on horizontal-edge image
        axes[1, 0].imshow(img_horz, cmap='gray')
        axes[1, 0].set_title('Input: Horizontal edge (top|bottom)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(KERNEL_HORIZONTAL_EDGE, cmap='RdBu')
        axes[1, 1].set_title('Kernel: Horizontal Edge')
        axes[1, 1].axis('off')
        
        out_h = conv2d_raw(img_horz, KERNEL_HORIZONTAL_EDGE)
        axes[1, 2].imshow(out_h, cmap='gray')
        axes[1, 2].set_title('Output: responds at boundary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('conv_example_output.png')
        print("\n[6] Saved visualization to conv_example_output.png")
        plt.close()
    except ImportError:
        print("\n[6] Install matplotlib to see visualizations: pip install matplotlib")


if __name__ == "__main__":
    main()

from typing import List

def conv2d_valid(image: List[List[float]], kernel: List[List[float]]) -> List[List[float]]:
    """
    2D convolution in DL style (cross-correlation): no kernel flip.
    No padding, stride=1.
    image: HxW
    kernel: kxk (assume square for this assignment)
    returns: (H-k+1) x (W-k+1)
    """
    H, W = len(image), len(image[0])
    k = len(kernel)
    assert all(len(row) == W for row in image), "Image rows must have equal length"
    assert all(len(row) == k for row in kernel), "Kernel must be k x k"
    assert H >= k and W >= k, "Kernel must be smaller than or equal to image"
    out_h, out_w = H - k + 1, W - k + 1
    out = [[0 for _ in range(out_w)] for _ in range(out_h)]

    for i in range(out_h):            # top-left corner of the sliding window
        for j in range(out_w):
            s = 0
            for u in range(k):
                for v in range(k):
                    s += kernel[u][v] * image[i + u][j + v]
            out[i][j] = s
    return out

if __name__ == "__main__":
    image_6x6 = [
        [1, 2, 3, 0, 1, 2],
        [0, 1, 2, 3, 1, 0],
        [1, 0, 1, 2, 2, 1],
        [2, 1, 0, 1, 3, 2],
        [0, 1, 2, 1, 0, 1],
        [1, 0, 1, 0, 1, 2],
    ]
    kernel_3x3 = [
        [0, 1, 0],
        [1, 0, -1],
        [0, -1, 0],
    ]
    out = conv2d_valid(image_6x6, kernel_3x3)
    for row in out:
        print(row)

import numpy as np

def cubic_bezier(P0, cp1, cp2, P1, t):
    """
    Compute a point on a cubic Bézier curve for a given t in [0,1].

    Args:
        P0 (tuple): Start point (x0, y0)
        cp1 (tuple): First control point
        cp2 (tuple): Second control point
        P1 (tuple): End point (x1, y1)
        t (float or np.ndarray): Curve parameter(s)

    Returns:
        np.ndarray: Interpolated point(s) on the curve
    """
    P0, cp1, cp2, P1 = map(np.array, [P0, cp1, cp2, P1])
    return ((1 - t) ** 3)[:, None] * P0 + \
           (3 * (1 - t) ** 2 * t)[:, None] * cp1 + \
           (3 * (1 - t) * t ** 2)[:, None] * cp2 + \
           (t ** 3)[:, None] * P1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 假设 A, B 是两条线段，每条有20个点
t = np.linspace(0, 2*np.pi, 20)
A = np.column_stack([t, np.sin(t)])         # A: y = sin(x)
B = np.column_stack([t, np.sin(t) + 1])     # B: y = sin(x) + 1

# 组合多边形的点：A 正向 + B 反向
polygon_points = np.vstack([A, B[::-1]])

# 方法1: plt.fill
plt.plot(A[:,0], A[:,1], 'r-', label='Line A')
plt.plot(B[:,0], B[:,1], 'b-', label='Line B')
plt.fill(polygon_points[:,0], polygon_points[:,1], color='lightblue', alpha=0.5)

# 方法2: Polygon Patch
# poly = Polygon(polygon_points, closed=True, facecolor='lightblue', alpha=0.5)
# plt.gca().add_patch(poly)

plt.legend()
plt.axis('equal')
plt.show()

from densify import densify

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)

def plot(pts, tri):
    plt.scatter([p[0] for p in tri]+[tri[0][0]] + [p[0] for p in pts], [p[1] for p in tri]+[tri[0][1]] + [p[1] for p in pts])
    # plt.plot([p[0] for p in tri]+[tri[0][0]], [p[1] for p in tri]+[tri[0][1]])
    # plt.scatter([p[0] for p in pts],[p[1] for p in pts])


def PointInsideTriangle2(pt,tri):
    '''checks if point pt(2) is inside triangle tri(3x2).'''
    a = 1/(-tri[1,1]*tri[2,0]+tri[0,1]*(-tri[1,0]+tri[2,0])+ \
        tri[0,0]*(tri[1,1]-tri[2,1])+tri[1,0]*tri[2,1])
    s = a*(tri[2,0]*tri[0,1]-tri[0,0]*tri[2,1]+(tri[2,1]-tri[0,1])*pt[0]+ \
        (tri[0,0]-tri[2,0])*pt[1])
    if s<0: return False
    else: t = a*(tri[0,0]*tri[1,1]-tri[1,0]*tri[0,1]+(tri[0,1]-tri[1,1])*pt[0]+ \
              (tri[1,0]-tri[0,0])*pt[1])
    return ((t>0) and (1-s-t>0))


# define the coordinates of our triangle corners
tri = np.array([
    [-2,1],
    [10,5],
    [4,9]
])

# pick a bunch of random points
num = 20
pts = np.random.uniform(tri.min(),tri.max(),num).reshape((num//2,2))

# let's find which ones lie inside the triangle
res = []
for p in pts:
    ans = PointInsideTriangle2(p, tri)
    res.append(ans)
res = np.array(res)

# numpy allows us to filter the points based on the above step
good = pts[res==True]
bad = pts[res==False]

# plot the good points inside the triangle

points = np.vstack((good, tri))
new_points, _ = densify(points, radius=1)
plt.scatter(points[:, 0], points[:, 1], s=10, c="blue")
plt.scatter(new_points[:, 0], new_points[:, 1], s=10, c="orange")
plt.show()

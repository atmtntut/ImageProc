from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
n = 5
xs = np.tile(np.linspace(8000, 20000, n), n).reshape(n, n)
print(xs)
ys = np.repeat(np.linspace(21000, 33000, n), n).reshape(n, n)
print(ys)
zs = np.tile(np.linspace(1100, 1160, n), n).reshape(n, n)
zs += np.random.rand(n**2).reshape(n, n)*10
print(zs)

ax.plot_wireframe(xs, ys, zs, rstride=1, cstride=1)
 
plt.show()

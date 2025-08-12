import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib import style
import random
import math




bodies_data = [
    { "name" : "Sun", "pos": [7, 13, 21], "v_i": [80, 6, 45], "mass": 100, "radius": 57 }, 
    { "name" : "Mercury", "pos": [0, 0, 0], "v_i": [0, 0, 0], "mass": 0, "radius": 0 }, 
    { "name" : "Venus", "pos": [0, 0, 0], "v_i": [0, 0, 0], "mass": 0, "radius": 0 }, 
                
                ]


# Earth's orbit around the sun

# Labels

# Definitions (meters)

AU = 149597870700
D_Sun = 1400000000
G = 6.67430e-11 
# declare Planet class

class Body: 

    bodies = []
    time = 0
    dt = 0.01
    T_FINAL = 3600 * 24 * 365 * 165

    def __init__(self, name, pos, v_i, mass, radius):
        self.name = name
        self.pos = np.array(pos) 
        self.v = np.array(v_i) 
        self.v_i = np.array(v_i)
        self.mass = mass
        self.radius = radius
        Body.bodies.append(self)
        

    # updates the acceleration for all bodies
    @staticmethod
    def compute_all_accel(r):
        accelerations = np.zeros((len(Body.bodies), 3))
        for i, body in enumerate(Body.bodies):
            accel = np.zeros(3) #resets every loop
            for j, other in enumerate(Body.bodies):
                if i != j:
                    r_vec = r[j] - r[i]
                    r_norm = np.linalg.norm(r_vec)
                    accel += (G * other.mass / r_norm**3) * r_vec
            accelerations[i] = accel
        return accelerations

    @staticmethod
    def rk4_step(dt):
        r0 = np.array([body.pos for body in Body.bodies])
        v0 = np.array([body.v_i for body in Body.bodies])

        k1r = v0
        k1v = Body.compute_all_accel(r0)

        k2r = v0 + (1/2) * dt * k1v
        k2v = Body.compute_all_accel(r0 + k1r * dt * (1/2))

        k3r = v0 + (1/2) * dt * k2v
        k3v = Body.compute_all_accel(r0 + (1/2) * k2r * dt)

        k4r = v0 + k3v * dt
        k4v = Body.compute_all_accel(r0 + k3r * dt)

        r_next = r0 + (k1r + 2*k2r + 2*k3r + k4r) * (dt/6)
        v_next = v0 + (k1v + 2*k2v + 2*k3v + k4v) * (dt/6)

        for i, body in enumerate(Body.bodies):
            body.pos = r_next[i]
            body.v_i = v_next[i]
        Body.time += dt

    def step_simulation(dt):
        Body.rk4_step(dt)
        return [body.pos.copy() for body in Body.bodies]

    def update_all(dt):
        snapshots = []
        Body.rk4_step(dt)
        snapshot = [body.pos.copy() for body in Body.bodies]
        snapshots.append(snapshot)

        Body.time += dt
        return snapshots


# initialize the planets
sun = Body("Sun", pos=[0, 0, 0], v_i=[0, 0, 0], mass=1.989e30, radius=6.9634e8)  # Sun radius approx 696,340 km

mercury = Body("Mercury", pos=[0.387 * AU, 0, 0], v_i=[0, 47360, 0], mass=3.301e23, radius=2.44e6)
venus = Body("Venus", pos=[0.723 * AU, 0, 0], v_i=[0, 35020, 0], mass=4.867e24, radius=6.052e6)
earth = Body("Earth", pos=[1.0 * AU, 0, 0], v_i=[0, 29784.7, 0], mass=5.972e24, radius=6.371e6)
mars = Body("Mars", pos=[1.524 * AU, 0, 0], v_i=[0, 24130, 0], mass=6.417e23, radius=3.39e6)

jupiter = Body("Jupiter", pos=[5.204 * AU, 0, 0], v_i=[0, 13070, 0], mass=1.898e27, radius=6.9911e7)
saturn = Body("Saturn", pos=[9.582 * AU, 0, 0], v_i=[0, 9690, 0], mass=5.683e26, radius=5.8232e7)
uranus = Body("Uranus", pos=[19.218 * AU, 0, 0], v_i=[0, 6810, 0], mass=8.681e25, radius=2.5362e7)
neptune = Body("Neptune", pos=[30.11 * AU, 0, 0], v_i=[0, 5430, 0], mass=1.024e26, radius=2.4622e7)


fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')


ax.set_xlim(-35 * AU, 35 * AU)
ax.set_ylim(-35 * AU, 35 * AU)
ax.set_zlim(-35 * AU, 35 * AU)
ax.set_xlabel("x, in meters")
ax.set_ylabel("y, in meters")
ax.set_zlabel("z, in meters")

ax.set_title("Solar System Orbit Animation")
ax.grid(True)
ax.set_aspect('equal')

colors = {
    "Sun": "yellow",
    "Mercury": "darkgray",
    "Venus": "orange",
    "Earth": "blue",
    "Mars": "red",
    "Jupiter": "brown",
    "Saturn": "gold",
    "Uranus": "lightblue",
    "Neptune": "darkblue",
}
scale_factor = 1e-6  
names = [body.name for body in Body.bodies]
sizes = [body.radius * scale_factor for body in Body.bodies]


#AI

positions_history = [[] for _ in Body.bodies]

points = []
for i, body in enumerate(Body.bodies):
    color = colors.get(body.name, 'black')
    x, y, z = body.pos
    scatters = ax.scatter([x], [y], [z], s=sizes[i], c=color, label=body.name)
    points.append(scatters)

trails = []
for i, name in enumerate(names):
    #plot returns list with line2D object, comma upacks and takes the object
    line, = ax.plot([], [], [], '-', color = colors.get(name, 'black'), alpha = 0.5,) 
    trails.append(line) #line is now line2D object

ax.legend()

def update(frame):
    Body.rk4_step(3600)  # advance simulation by 1 hour (same as before)

    for i, body in enumerate(Body.bodies):
        x, y, z = body.pos
        points[i]._offsets3d = ([x], [y], [z])

        positions_history[i].append(body.pos.copy())
        trail_x = [pos[0] for pos in positions_history[i]]
        trail_y = [pos[1] for pos in positions_history[i]]
        trail_z = [pos[2] for pos in positions_history[i]]

        trails[i].set_data(trail_x, trail_y)
        trails[i].set_3d_properties(trail_z)

    return points + trails

ani = FuncAnimation(fig, update, frames=range(0, 10000), interval=10, blit=False)
# ani.save("3DSS_test2.mp4", fps=15, dpi=72)

plt.show()
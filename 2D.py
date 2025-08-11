import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from matplotlib.animation import FuncAnimation
import random


AU = 149597870700
D_Sun = 1400000000
G = 6.67430e-11 
# declare Planet class

class Body: 

    bodies = []
    time = 0
    T_FINAL = 3600 * 24 * 365.25 * 10


    def __init__(self, name, pos, v_i, mass, radius):
        self.name = name
        self.pos = np.array(pos) 
        self.v = np.array(v_i) 
        self.v_i = np.array(v_i)
        self.mass = mass
        self.radius = radius
        Body.bodies.append(self)
        

    # calculates the total 2-dimensional acceleration for a single body at one time-step
    # ODE #1
    def a(self):
        acceleration_2d = np.zeros(2)
        for other in Body.bodies:
            if other is not self:
                r_vec = other.pos - self.pos
                r_norm = np.linalg.norm(r_vec)
                acceleration_2d += (G * other.mass / r_norm**3) * r_vec
        return acceleration_2d


    # updates the acceleration for all bodies
    @staticmethod
    def compute_all_accel(r):
        n = len(Body.bodies)
        accelerations = np.zeros((len(Body.bodies), 2))
        for i in range(n):
            accel = np.zeros(2) #resets every loop
            for j, other in enumerate(Body.bodies):
                if i != j:
                    r_vec = r[j] - r[i]
                    r_norm = np.linalg.norm(r_vec)
                    accel += (G * other.mass / r_norm**3) * r_vec
            accelerations[i] = accel
        #print("this is acceleration", accelerations)
        return accelerations
    
    '''
    @staticmethod
    def update_all_velocity(dt, update_instances = True):
        position_0 = np.array([x.pos.copy() for x in Body.bodies])
        velocity_0 = np.array([x.v_i.copy() for x in Body.bodies])
    
        k1 = Body.compute_all_accel(position_0) * dt
        k2 = Body.compute_all_accel(position_0 + (velocity_0 + k1/2) * dt/2) * dt
        k3 = Body.compute_all_accel(position_0 + (velocity_0 + k2/2) * dt/2) * dt
        k4 = Body.compute_all_accel(position_0 + (velocity_0 + k3) * dt) * dt

        velocity_new = velocity_0 + (1/6) * (k1 + 2*k2 + 2*k3 + 2*k4)

        if update_instances:
            for i, body in enumerate(Body.bodies):
                body.v_i = velocity_new[i]

        #print("this is velocity:", velocity_new)
    
    @staticmethod
    def velocity_at(t, r):
        velocity_0 = np.array([x.v_i.copy() for x  in Body.bodies])
        
        if t == Body.time: 
            return velocity_0
        else: 
            return velocity_0 + Body.compute_all_accel(r) * (t - Body.time)


    @staticmethod
    def update_all_position(dt, update_instances = True):
        position_0 = np.array([x.pos.copy() for x in Body.bodies])

        k1 = Body.velocity_at(Body.time, position_0)
        k2 = Body.velocity_at(Body.time + dt/2, position_0 + k1 * (dt/2))
        k3 = Body.velocity_at(Body.time + dt/2, position_0 + k2 * (dt/2))
        k4 = Body.velocity_at(Body.time + dt, position_0 + k3 * dt)

        position_new = position_0 + (dt/6) * (k1 + 2*k2 + 2*k3 + 2*k4)

        if update_instances:
            for i, body in enumerate(Body.bodies):
                body.pos =  position_new[i]
        
        return position_new
    '''
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


        #under the huge while loop computer everything to keep everything synchronized
        # do acceleration and then velocity and then position within the while loop( i think)


    '''
    @staticmethod
    def update_all(dt):
        positions_over_time = []
        while Body.time < Body.T_FINAL:
            Body.update_all_velocity(dt, True)
            Body.update_all_position(dt, True)

            snapshot = [body.pos.copy() for body in Body.bodies]
            positions_over_time.append(snapshot)    

            Body.time += dt
        
        return positions_over_time 
    '''
    @staticmethod
    def update_all(dt):
        snapshots = []
        while Body.time < Body.T_FINAL:
            Body.rk4_step(dt)
            snapshot = [body.pos.copy() for body in Body.bodies]
            snapshots.append(snapshot)

            Body.time += dt
        return snapshots

# reset
Body.bodies = []
Body.time = 0
Body.positions_over_time = []

sun = Body("Sun", pos=[0, 0], v_i=[0, 0], mass=1.989e30, radius=D_Sun/2)

mercury = Body("Mercury", pos=[0.387 * AU, 0], v_i=[0, 47_360], mass=3.301e23, radius=2.44e6)

venus = Body("Venus", pos=[0.723 * AU, 0], v_i=[0, 35_020], mass=4.867e24, radius=6.052e6)

earth = Body("Earth", pos=[AU, 0], v_i=[0, 29_784.7], mass=5.972e24, radius=6.371e6)

'''

mars = Body("Mars", pos=[1.524 * AU, 0], v_i=[0, 24_130], mass=6.417e23, radius=3.390e6)

jupiter = Body("Jupiter", pos=[5.204 * AU, 0], v_i=[0, 13_070], mass=1.898e27, radius=6.9911e7)

saturn = Body("Saturn", pos=[9.582 * AU, 0], v_i=[0, 9_690], mass=5.683e26, radius=5.8232e7)

uranus = Body("Uranus", pos=[19.218 * AU, 0], v_i=[0, 6_810], mass=8.681e25, radius=2.5362e7)
'''



# [[[1,2,3], [3,4,5], [[5,6,7], [7,8,9]], [[9,10,11], [11,12,13]]]
stored_positions = Body.update_all(3600)
x_values = [[None for _ in Body.bodies] for _ in range(len(stored_positions))]
# [[[1], [3], [[5], [7]], [[9], [11]]]
y_values = [[None for _ in Body.bodies] for _ in range(len(stored_positions))]

for i, body in enumerate(stored_positions):
    for j, skibidi in enumerate(body):
        x_values[i][j] = skibidi[0]
        y_values[i][j] = skibidi[1]



#for i in range(5):
    #print(f"{x_values[i][1]}, {y_values[i][1]}") #time, body


#AI rewrite all of this
fig, ax = plt.subplots(figsize=(10, 10))

ax.set_xlim(-35 * AU, 35 * AU)
ax.set_ylim(-35 * AU, 35 * AU)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Solar System Orbit Animation")
ax.grid(True)
ax.set_aspect('equal')

# Color map and body names
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
names = [body.name for body in Body.bodies]

# Initialize planet dots
points = []
for i, name in enumerate(names):
    color = colors.get(name, 'black')
    point, = ax.plot([], [], 'o', label=name, color=color)
    points.append(point)

ax.legend()

# Update function for each animation frame
def update(frame):
    for i, point in enumerate(points):
        x = x_values[frame][i]
        y = y_values[frame][i]
        point.set_data([x], [y])  # wrap in lists
    return points

# Create and show the animation
ani = FuncAnimation(fig, update, frames=range(0, len(x_values), 10), interval=50, blit=True)
ani.save("SS_test3.mp4", fps=15, dpi=72)


plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from collision import Particle, Simulation

class PeriodicParticle(Particle):
    def overlaps(self, other):
        """Does the circle of this Particle overlap that of other?"""

        total = 0.
        dx = abs(self.x - other.x)
        dx = min(dx, 1-dx)
        dy = abs(self.y - other.y)
        dy = min(dy, 1-dy)
        return np.hypot(dx, dy) < self.radius + other.radius

    def draw(self, ax):
        """Add this Particle's Circle patch to the Matplotlib Axes ax."""

        circle = Circle(xy=self.r, radius=self.radius, **self.styles)
        ax.add_patch(circle)

        if self.x + self.radius > 1:
            ax.add_patch(Circle(xy=(self.x-1, self.y),
                                radius=self.radius, **self.styles))
        if self.x - self.radius < 0:
            ax.add_patch(Circle(xy=(1+self.x, self.y),
                                radius=self.radius, **self.styles))
        if self.y + self.radius > 1:
            ax.add_patch(Circle(xy=(self.x, self.y-1),
                                radius=self.radius, **self.styles))
        if self.y - self.radius < 0:
            ax.add_patch(Circle(xy=(self.x, 1+self.y),
                                radius=self.radius, **self.styles))
        return circle

    def advance(self, dt):
        """Advance the Particle's position forward in time by dt."""

        self.x = (self.x + self.vx * dt) % 1
        self.y = (self.y + self.vy * dt) % 1

class PeriodicSimulation(Simulation):
    """A class for a simple hard-circle molecular dynamics simulation.

    The simulation is carried out on a square domain: 0 <= x < 1, 0 <= y < 1.

    """

    ParticleClass = PeriodicParticle

    def init_particles(self, n, radius, styles):
        self.n = n
        # First place the large, stationary particle.
        p0 = self.ParticleClass(0.5, 0.5, 0, 0, 0.1,
                {'edgecolor': 'C2', 'facecolor': 'C2'})
        self.particles = [p0, ]
        # Now place the other, smaller, moving particles.
        for i in range(n-1):
            # Try to find a random initial position for this particle.
            while not self.place_particle(radius, styles):
                pass

    def handle_boundary_collisions(self, p):
        pass

    def advance_animation(self):
        """Advance the animation by self.dt."""

        # Blitting would be a bit complicated because circle patches come and
        # go as a circle crosses the periodic boundaries, so we make life
        # easy for ourselves and redraw the whole Axes object for each frame.
        self.advance()
        self.ax.clear()
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])
        for particle in self.particles:
            particle.draw(self.ax)
        return

    def do_animation(self, save=False, filename='collision.mp4'):
        """Set up and carry out the animation of the molecular dynamics."""

        self.setup_animation()
        self.init()
        anim = animation.FuncAnimation(self.fig, self.animate,
                               frames=100, interval=1, blit=False)
        self.save_or_show_animation(anim, save, filename)

if __name__ == '__main__':
    # 99 small particles and one large one.
    nparticles = 100
    small_particle_radius = 0.02
    styles = {'edgecolor': 'C0', 'linewidth': 2, 'fill': None}
    sim = PeriodicSimulation(nparticles, small_particle_radius, styles)
    # Despite being bigger, set the mass of the large particle to be the same
    # as the small ones so it gains a bit of momentum in the collisions
    sim.particles[0].mass = small_particle_radius**2
    sim.dt = 0.02
    sim.do_animation(save=False)
#    sim.do_animation(save=True, filename='brownian.mp4')


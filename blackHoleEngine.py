import numpy as np

class BlackHoleEngine:
    """Core physics engine for black hole simulation with Barnes-Hut optimization for N-body."""

    G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
    c = 299792458    # Speed of light (m/s)
    SOLAR_MASS = 1.989e30  # kg

    def __init__(self, mass, position, velocity):
        self.mass = mass * self.SOLAR_MASS  # Convert solar masses to kg
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.bodies = []  # List of orbiting bodies

    def calculate_schwarzschild_radius(self):
        """Calculate the Schwarzschild radius (event horizon) of the black hole."""
        return 2 * self.G * self.mass / (self.c ** 2)

    def calculate_photon_sphere_radius(self):
        """Calculate the photon sphere radius (1.5 * Rs)."""
        return 1.5 * self.calculate_schwarzschild_radius()

    def calculate_isco_radius(self):
        """Calculate innermost stable circular orbit (3 * Rs for Schwarzschild)."""
        return 3 * self.calculate_schwarzschild_radius()

    def add_body(self, mass, position, velocity):
        """Add an orbiting body (planet, star, etc.) to the simulation."""
        body = {
            'mass': mass,
            'position': np.array(position, dtype=np.float64),
            'velocity': np.array(velocity, dtype=np.float64),
            'acceleration': np.zeros(len(position), dtype=np.float64)
        }
        self.bodies.append(body)
        return len(self.bodies) - 1  # Return body index


class BarnesHutNode:
    """Octree/Quadtree node for Barnes-Hut algorithm."""

    def __init__(self, center, size, dimension=3):
        self.center = np.array(center, dtype=np.float64)
        self.size = size
        self.dimension = dimension  # 2D or 3D
        self.total_mass = 0.0
        self.center_of_mass = np.zeros(dimension, dtype=np.float64)
        self.body = None  # Single body if leaf node
        self.children = []  # Child nodes if internal node

    def is_leaf(self):
        return len(self.children) == 0

    def is_external(self):
        return self.body is not None and len(self.children) == 0

    def insert(self, body):
        """Insert a body into the tree."""
        # If node is empty, add body here
        if self.total_mass == 0:
            self.body = body
            self.total_mass = body['mass']
            self.center_of_mass = body['position'].copy()
            return

        # If node is external (has one body), subdivide
        if self.is_external():
            old_body = self.body
            self.body = None
            self._subdivide()
            self._insert_to_child(old_body)

        # Update center of mass
        total_mass_new = self.total_mass + body['mass']
        self.center_of_mass = (self.center_of_mass * self.total_mass +
                               body['position'] * body['mass']) / total_mass_new
        self.total_mass = total_mass_new

        # Insert to appropriate child
        self._insert_to_child(body)

    def _subdivide(self):
        """Create child nodes."""
        half_size = self.size / 2
        quarter_size = self.size / 4

        if self.dimension == 2:
            # Quadtree (4 children)
            offsets = [
                [-quarter_size, -quarter_size],
                [quarter_size, -quarter_size],
                [-quarter_size, quarter_size],
                [quarter_size, quarter_size]
            ]
        else:
            # Octree (8 children)
            offsets = [
                [-quarter_size, -quarter_size, -quarter_size],
                [quarter_size, -quarter_size, -quarter_size],
                [-quarter_size, quarter_size, -quarter_size],
                [quarter_size, quarter_size, -quarter_size],
                [-quarter_size, -quarter_size, quarter_size],
                [quarter_size, -quarter_size, quarter_size],
                [-quarter_size, quarter_size, quarter_size],
                [quarter_size, quarter_size, quarter_size]
            ]

        for offset in offsets:
            child_center = self.center + np.array(offset)
            self.children.append(BarnesHutNode(child_center, half_size, self.dimension))

    def _insert_to_child(self, body):
        """Insert body into appropriate child."""
        for child in self.children:
            if self._contains(child, body['position']):
                child.insert(body)
                return

    def _contains(self, node, position):
        """Check if position is within node bounds."""
        half_size = node.size / 2
        for i in range(self.dimension):
            if abs(position[i] - node.center[i]) > half_size:
                return False
        return True

    def calculate_force(self, body, theta=0.5):
        """
        Calculate gravitational force on body using Barnes-Hut algorithm.
        theta: opening angle threshold (lower = more accurate, higher = faster)
        """
        # If node is empty, no force
        if self.total_mass == 0:
            return np.zeros(self.dimension, dtype=np.float64)

        # Vector from body to center of mass
        diff = self.center_of_mass - body['position']
        distance = np.linalg.norm(diff)

        # Avoid self-interaction and division by zero
        if distance < 1e-10:
            return np.zeros(self.dimension, dtype=np.float64)

        # If node is sufficiently far away, treat as single body
        if self.is_external() or (self.size / distance < theta):
            # F = G * m1 * m2 / r^2, direction towards center of mass
            force_magnitude = (BlackHoleEngine.G * body['mass'] *
                              self.total_mass / (distance ** 2))
            force = force_magnitude * (diff / distance)
            return force

        # Otherwise, recursively calculate force from children
        force = np.zeros(self.dimension, dtype=np.float64)
        for child in self.children:
            force += child.calculate_force(body, theta)

        return force


class NBodySimulation:
    """N-body simulation using Barnes-Hut algorithm."""

    def __init__(self, dimension=3, theta=0.5):
        self.dimension = dimension
        self.theta = theta  # Barnes-Hut opening angle
        self.bodies = []
        self.time = 0.0

    def add_body(self, mass, position, velocity):
        """Add a body to the simulation."""
        body = {
            'mass': mass,
            'position': np.array(position, dtype=np.float64),
            'velocity': np.array(velocity, dtype=np.float64),
            'acceleration': np.zeros(self.dimension, dtype=np.float64)
        }
        self.bodies.append(body)
        return len(self.bodies) - 1

    def build_tree(self):
        """Build Barnes-Hut tree from current body positions."""
        if len(self.bodies) == 0:
            return None

        # Find bounding box
        positions = np.array([b['position'] for b in self.bodies])
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)

        # Create root node with some padding
        center = (min_pos + max_pos) / 2
        size = np.max(max_pos - min_pos) * 1.5

        root = BarnesHutNode(center, size, self.dimension)

        # Insert all bodies
        for body in self.bodies:
            root.insert(body)

        return root

    def update(self, dt):
        """Update simulation by one time step using Velocity Verlet integration."""
        # Build tree
        tree = self.build_tree()
        if tree is None:
            return

        # Calculate forces for all bodies
        for body in self.bodies:
            force = tree.calculate_force(body, self.theta)
            body['acceleration'] = force / body['mass']

        # Update positions and velocities (Velocity Verlet)
        for body in self.bodies:
            # Update position: x += v*dt + 0.5*a*dt^2
            body['position'] += body['velocity'] * dt + 0.5 * body['acceleration'] * dt**2

            # Store old acceleration
            old_acceleration = body['acceleration'].copy()

        # Recalculate forces with new positions
        tree = self.build_tree()
        for body in self.bodies:
            force = tree.calculate_force(body, self.theta)
            new_acceleration = force / body['mass']

            # Update velocity: v += 0.5*(a_old + a_new)*dt
            body['velocity'] += 0.5 * (old_acceleration + new_acceleration) * dt
            body['acceleration'] = new_acceleration

        self.time += dt
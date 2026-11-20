import pygame
import numpy as np
from blackHoleEngine import BlackHoleEngine

class PygameRenderer:
    def __init__(self, engine, width = 1600, height = 900):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Black Hole Simulation - Spacetime Visualization")
        self.clock = pygame.time.Clock()
        self.running = True
        self.scaleFactor = 3
        self.engine = engine
        self.width = width
        self.height = height

        # Enhanced color palette
        self.Black = (0, 0, 0)
        self.White = (255, 255, 255)
        self.Space = (5, 5, 15)  # Darker space
        self.GridColor = (60, 60, 80)
        self.AccretionOrange = (255, 140, 0)
        self.AccretionYellow = (255, 200, 50)
        self.PhotonSphere = (100, 100, 150)

        # Camera controls - proper initial view
        self.camera_angle_x = 0.5  # Positive tilt to view from above
        self.camera_angle_y = 0.0  # No rotation
        self.zoom = 1.0  # Normal zoom

        # No rays for cleaner visualization
        self.rays = []

    def world_to_screen(self, position):
        positionX = np.array(position[0]) * self.scaleFactor * self.zoom + self.width // 2
        positionY = self.height // 2 - np.array(position[1]) * self.scaleFactor * self.zoom
        return (int(positionX), int(positionY))

    def project_3d_to_2d(self, x, y, z):
        # Apply 3D rotation for perspective
        # Rotate around X axis (tilt) - note Y is vertical in our system
        y_rot_x = y * np.cos(self.camera_angle_x) - z * np.sin(self.camera_angle_x)
        z_rot_x = y * np.sin(self.camera_angle_x) + z * np.cos(self.camera_angle_x)

        # Rotate around Y axis (spin)
        x_rot_y = x * np.cos(self.camera_angle_y) - z_rot_x * np.sin(self.camera_angle_y)
        z_final = x * np.sin(self.camera_angle_y) + z_rot_x * np.cos(self.camera_angle_y)

        # Enhanced perspective projection
        camera_distance = 1200
        fov = 1000

        # Prevent division by zero
        if (camera_distance - z_final) > 0:
            perspective = fov / (camera_distance - z_final)
        else:
            perspective = fov / camera_distance

        screen_x = x_rot_y * perspective * self.zoom + self.width / 2
        # Invert Y for screen coordinates (screen Y increases downward)
        screen_y = -y_rot_x * perspective * self.zoom + self.height / 2

        return (int(screen_x), int(screen_y))

    def draw_spacetime_grid(self):
        grid_size = 80  # Even denser grid for smoother curves
        spacing = 12    # Wider spacing for broader grid
        rs_km = self.engine.calculate_schwarzschild_radius() / 1000

        # Pre-calculate all vertices with gravitational lensing
        vertices = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                # Center the grid
                x_original = (i - grid_size/2) * spacing
                z_original = (j - grid_size/2) * spacing

                # No lensing distortion on grid itself - keep grid regular
                x = x_original
                z = z_original

                # Calculate distance from black hole
                dist = np.sqrt(x**2 + z**2)

                # Create the gravity well visualization
                # Grid warps DOWN into the black hole, creating a funnel
                if dist > rs_km * 1.5:  # Outside event horizon
                    # Smooth warping downward
                    normalized_dist = (dist - rs_km) / rs_km
                    if normalized_dist < 10:
                        # Strong warping near black hole
                        # Use a smoother function that starts at 0 at the edge
                        y = -6 * rs_km * np.exp(-normalized_dist / 2) / (1 + normalized_dist * 0.3)
                    else:
                        # Far field - approaches flat (y=0)
                        y = -6 * rs_km * np.exp(-normalized_dist / 2) / (1 + normalized_dist * 0.3)

                    # Ensure smooth transition to flat space
                    if normalized_dist > 5:
                        y = y * np.exp(-(normalized_dist - 5) / 5)
                else:
                    # Very close to event horizon - steep drop
                    # This creates the visual "hole" effect
                    y = -10 * rs_km

                row.append((x, y, z))
            vertices.append(row)

        # Draw horizontal grid lines with gradient coloring
        for i in range(grid_size):
            points = []
            skip_indices = []
            for j in range(grid_size):
                vertex = vertices[i][j]
                x, y, z = vertex
                dist = np.sqrt(x**2 + z**2)

                # Skip drawing lines through the event horizon
                if dist < rs_km * 1.2:
                    skip_indices.append(j)

                screen_point = self.project_3d_to_2d(vertex[0], vertex[1], vertex[2])
                points.append(screen_point)

            if len(points) > 1:
                # Draw with depth-based color, skipping event horizon
                for k in range(len(points) - 1):
                    # Don't draw lines through the black hole
                    if k in skip_indices or (k+1) in skip_indices:
                        continue

                    # Calculate color based on depth
                    depth = abs(vertices[i][k][1])
                    max_depth = 80 * rs_km
                    color_factor = min(1.0, depth / max_depth)

                    # Gradient from blue-gray to darker
                    base_color = 40
                    color_range = 50
                    intensity = int(base_color + (1 - color_factor) * color_range)
                    color = (intensity, intensity, intensity + 15)

                    pygame.draw.line(self.screen, color, points[k], points[k+1], 1)

        # Draw vertical grid lines
        for j in range(grid_size):
            points = []
            skip_indices = []
            for i in range(grid_size):
                vertex = vertices[i][j]
                x, y, z = vertex
                dist = np.sqrt(x**2 + z**2)

                # Skip drawing lines through the event horizon
                if dist < rs_km * 1.2:
                    skip_indices.append(i)

                screen_point = self.project_3d_to_2d(vertex[0], vertex[1], vertex[2])
                points.append(screen_point)

            if len(points) > 1:
                for k in range(len(points) - 1):
                    # Don't draw lines through the black hole
                    if k in skip_indices or (k+1) in skip_indices:
                        continue

                    # Calculate color based on depth
                    depth = abs(vertices[k][j][1])
                    max_depth = 80 * rs_km
                    color_factor = min(1.0, depth / max_depth)

                    base_color = 40
                    color_range = 50
                    intensity = int(base_color + (1 - color_factor) * color_range)
                    color = (intensity, intensity, intensity + 15)

                    pygame.draw.line(self.screen, color, points[k], points[k+1], 1)

    def draw_gravitational_lensing_effect(self):
        """Draw Einstein ring and gravitational lensing effects"""
        rs_km = self.engine.calculate_schwarzschild_radius() / 1000

        # Draw Einstein rings at different radii
        # These show where light paths are strongly bent
        einstein_radii = [1.5 * rs_km, 2.6 * rs_km, 3.5 * rs_km]

        for i, radius in enumerate(einstein_radii):
            num_points = 100
            points = []

            for j in range(num_points):
                angle = 2 * np.pi * j / num_points
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)

                # Einstein rings appear slightly above the plane due to lensing
                y = 0.5 * rs_km * (1 - i * 0.3)

                # Apply grid warping for consistency
                dist = np.sqrt(x**2 + z**2)
                if dist > rs_km:
                    normalized_dist = (dist - rs_km) / rs_km
                    y_warp = -8 * rs_km / (1 + normalized_dist * 0.5)
                    damping = np.exp(-normalized_dist / 3)
                    y = y + y_warp * damping * 0.3  # Partial warping for rings

                screen_point = self.project_3d_to_2d(x, y, z)
                points.append(screen_point)

            if len(points) > 2:
                # Fading effect for outer rings
                alpha = 80 - i * 20
                color = (alpha, alpha, alpha + 20)
                pygame.draw.lines(self.screen, color, True, points, 1)

    def draw_photon_sphere(self):
        """Draw the photon sphere at 1.5 * Schwarzschild radius"""
        rs_km = self.engine.calculate_schwarzschild_radius() / 1000
        photon_radius = 1.5 * rs_km

        # Draw a faint circle to indicate the photon sphere
        num_points = 100
        points = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = photon_radius * np.cos(angle)
            z = photon_radius * np.sin(angle)
            # Match grid warping formula
            dist = photon_radius
            if dist > rs_km:
                normalized_dist = (dist - rs_km) / rs_km
                y = -8 * rs_km / (1 + normalized_dist * 0.5)
                damping = np.exp(-normalized_dist / 3)
                y = y * damping
            else:
                y = -15 * rs_km

            screen_point = self.project_3d_to_2d(x, y, z)
            points.append(screen_point)

        if len(points) > 2:
            pygame.draw.lines(self.screen, self.PhotonSphere, True, points, 2)

    def draw_accretion_disk(self):
        """Draw accretion disk with gravitational lensing"""
        rs_km = self.engine.calculate_schwarzschild_radius() / 1000
        inner_radius = 3 * rs_km
        outer_radius = 15 * rs_km

        # Draw the accretion disk in the XZ plane (y=0)
        num_rings = 20

        for ring in range(num_rings):
            radius = inner_radius + (outer_radius - inner_radius) * ring / num_rings
            num_points = 80

            # Points for the main disk
            main_disk_points = []
            # Points for the gravitationally lensed images
            lensed_upper_points = []  # Halo above
            lensed_lower_points = []  # Secondary image below

            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)

                # Accretion disk is relatively flat, with only slight warping
                dist = np.sqrt(x**2 + z**2)
                if dist > rs_km * 3:  # Disk starts at 3x Schwarzschild radius
                    # Disk is mostly flat with very gentle warping
                    # Only slight dip near the inner edge
                    normalized_dist = (dist - rs_km * 3) / rs_km
                    y_disk = -0.5 * rs_km * np.exp(-normalized_dist / 2)
                else:
                    # No disk inside 3x Schwarzschild radius (inner edge)
                    continue  # Skip this point

                # Add main disk point
                screen_point = self.project_3d_to_2d(x, y_disk, z)
                main_disk_points.append(screen_point)

                # GRAVITATIONAL LENSING - The key insight:
                # We see the SAME disk from MULTIPLE viewpoints due to bent light
                # The "halo" is the FAR SIDE of the disk bent into view

                if dist > rs_km * 3:  # Disk outside innermost stable orbit
                    # Calculate angle around the black hole
                    theta = np.arctan2(z, x)

                    # UPPER LENSED IMAGE: Far side of disk appears above
                    # This creates the upper arc of the "halo"
                    # Light from z < 0 (back) bends over the top to reach us
                    if True:  # All points contribute, but back side is strongest
                        # Lensing strength depends on distance from black hole
                        lensing_strength = 2.5 * rs_km / dist

                        # Back side (z < 0) appears most strongly
                        # Front side (z > 0) appears weakly
                        back_factor = 1.0 - z / (dist + 0.1)  # Smooth transition

                        # Height of lensed image above black hole
                        y_lensed_up = 4 * rs_km * lensing_strength * back_factor

                        # The image wraps around - maintain x coordinate
                        # But compress in z to create the arc shape
                        x_lensed = x
                        z_lensed = -dist * 0.3  # Projects to a consistent radius

                        screen_point_up = self.project_3d_to_2d(x_lensed, y_lensed_up, z_lensed)
                        lensed_upper_points.append(screen_point_up)

                    # LOWER LENSED IMAGE: Far side also appears below (dimmer)
                    if dist < rs_km * 7:  # Closer orbits have stronger lower image
                        back_factor = 1.0 - z / (dist + 0.1)
                        y_lensed_down = -2.5 * rs_km * lensing_strength * back_factor

                        x_lensed = x
                        z_lensed = -dist * 0.25

                        screen_point_down = self.project_3d_to_2d(x_lensed, y_lensed_down, z_lensed)
                        lensed_lower_points.append(screen_point_down)

            # Color gradient from hot (inner) to cooler (outer)
            color_factor = 1.0 - ring / num_rings
            red = int(255 * color_factor)
            green = int(140 * color_factor + 60 * (1 - color_factor))
            blue = int(50 * (1 - color_factor))
            color = (red, green, blue)

            # Draw main disk
            if len(main_disk_points) > 1:
                for k in range(len(main_disk_points) - 1):
                    pygame.draw.line(self.screen, color, main_disk_points[k], main_disk_points[k+1], 2)

            # Draw gravitationally lensed image (the "halo" above the black hole)
            if len(lensed_upper_points) > 1:
                # Lensed image is slightly dimmer and has a different color due to redshift
                lensed_color = (int(red * 0.7), int(green * 0.6), int(blue * 0.5))
                for k in range(len(lensed_upper_points) - 1):
                    pygame.draw.line(self.screen, lensed_color, lensed_upper_points[k], lensed_upper_points[k+1], 2)

            # Draw secondary lensed image below (fainter)
            if len(lensed_lower_points) > 1:
                lensed_color_lower = (int(red * 0.4), int(green * 0.35), int(blue * 0.3))
                for k in range(len(lensed_lower_points) - 1):
                    pygame.draw.line(self.screen, lensed_color_lower, lensed_lower_points[k], lensed_lower_points[k+1], 1)

    def draw_black_hole(self):
        """Draw the event horizon"""
        world_pos = self.engine.position
        rs_km = self.engine.calculate_schwarzschild_radius() / 1000

        # Black hole sits at the center of the coordinate system
        # The grid warps around it, but the hole itself is at y=0
        y_warped = 0  # At the plane center
        screen_pos = self.project_3d_to_2d(world_pos[0], y_warped, world_pos[1])

        # Draw the black hole
        radius = int(rs_km * self.scaleFactor * self.zoom * 1.2)  # Slightly larger for visibility
        if radius < 1:
            radius = 1

        # Draw with a slight glow effect
        for i in range(3):
            glow_radius = radius + (3 - i) * 2
            alpha = 30 - i * 10
            color = (alpha, alpha, alpha + 10)
            pygame.draw.circle(self.screen, color, screen_pos, glow_radius, 1)

        # Draw the black center
        pygame.draw.circle(self.screen, self.Black, screen_pos, radius)

    def draw_rays(self):
        """Draw light rays with improved visualization"""
        for ray in self.rays:
            if hasattr(ray, 'path') and len(ray.path) > 1:
                screen_points = []
                for pos in ray.path:
                    # Convert ray position to 3D and apply warping
                    x, y = pos[0], pos[1]
                    dist = np.sqrt(x**2 + y**2)
                    rs_km = self.engine.calculate_schwarzschild_radius() / 1000

                    # Match grid warping for consistency
                    if dist > rs_km:
                        normalized_dist = (dist - rs_km) / rs_km
                        if normalized_dist < 10:
                            z = -8 * rs_km / (1 + normalized_dist * 0.5)
                            damping = np.exp(-normalized_dist / 3)
                            z = z * damping
                        else:
                            z = -8 * rs_km * np.exp(-normalized_dist / 3) / (1 + normalized_dist * 0.5)
                    else:
                        z = -15 * rs_km

                    screen_point = self.project_3d_to_2d(x, z, y)
                    screen_points.append(screen_point)

                if len(screen_points) > 1:
                    # Draw with fading effect
                    for i in range(len(screen_points) - 1):
                        alpha = max(0.2, 1.0 - i / len(screen_points))
                        intensity = int(255 * alpha)
                        color = (intensity, intensity, intensity)
                        pygame.draw.line(self.screen, color, screen_points[i], screen_points[i+1], 1)

    def handle_input(self):
        """Handle keyboard input for camera control"""
        keys = pygame.key.get_pressed()

        # Camera rotation
        if keys[pygame.K_LEFT]:
            self.camera_angle_y -= 0.02
        if keys[pygame.K_RIGHT]:
            self.camera_angle_y += 0.02
        if keys[pygame.K_UP]:
            self.camera_angle_x -= 0.02
        if keys[pygame.K_DOWN]:
            self.camera_angle_x += 0.02

        # Zoom
        if keys[pygame.K_z]:
            self.zoom *= 1.02
        if keys[pygame.K_x]:
            self.zoom *= 0.98

        # Reset view
        if keys[pygame.K_r]:
            self.camera_angle_x = 0.5
            self.camera_angle_y = 0.0
            self.zoom = 1.0

    def draw_info(self):
        """Draw information overlay"""
        font = pygame.font.Font(None, 24)
        info_lines = [
            f"Black Hole Mass: {self.engine.mass} Solar Masses",
            f"Schwarzschild Radius: {self.engine.calculate_schwarzschild_radius()/1000:.1f} km",
            "Controls: Arrow Keys (rotate), Z/X (zoom), R (reset)",
            f"Camera: Tilt={self.camera_angle_x:.2f}, Rotation={self.camera_angle_y:.2f}, Zoom={self.zoom:.2f}"
        ]

        y_offset = 10
        for line in info_lines:
            text = font.render(line, True, (150, 150, 170))
            self.screen.blit(text, (10, y_offset))
            y_offset += 25

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            # Handle continuous input
            self.handle_input()

            # Clear screen
            self.screen.fill(self.Space)

            # Draw everything in correct order
            self.draw_spacetime_grid()
            self.draw_gravitational_lensing_effect()  # Add lensing visualization
            self.draw_accretion_disk()  # Enable accretion disk
            self.draw_black_hole()
            self.draw_info()

            # Update display
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
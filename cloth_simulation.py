import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import time

class Particle:
    """Represents a single cloth particle with physics"""
    def __init__(self, x, y, z, mass=1.0):
        self.position = np.array([x, y, z], dtype=float)
        self.old_position = np.array([x, y, z], dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.force = np.array([0.0, 0.0, 0.0], dtype=float)
        self.mass = mass
        self.pinned = False
        self.is_cut = False
    
    def apply_force(self, force):
        """Apply force to particle"""
        if not self.pinned:
            self.force += force
    
    def update(self, dt, damping=0.99):
        """Update particle position using Verlet integration"""
        if self.pinned or self.is_cut:
            self.force = np.array([0.0, 0.0, 0.0])
            return
        
        # Verlet integration
        acceleration = self.force / self.mass
        new_position = self.position + (self.position - self.old_position) * damping + acceleration * dt * dt
        
        self.old_position = self.position.copy()
        self.position = new_position
        self.velocity = (self.position - self.old_position) / dt
        
        # Reset force
        self.force = np.array([0.0, 0.0, 0.0])


class Constraint:
    """Spring constraint between two particles"""
    def __init__(self, p1, p2, stiffness=0.99):
        self.p1 = p1
        self.p2 = p2
        self.rest_length = np.linalg.norm(p1.position - p2.position)
        self.stiffness = stiffness
        self.active = True
    
    def satisfy(self):
        """Enforce constraint using position-based dynamics"""
        if not self.active or self.p1.is_cut or self.p2.is_cut:
            return
        
        if self.p1.pinned and self.p2.pinned:
            return
        
        delta = self.p2.position - self.p1.position
        current_length = np.linalg.norm(delta)
        
        if current_length < 0.0001:
            return
        
        diff = (current_length - self.rest_length) / current_length
        correction = delta * diff * self.stiffness
        
        if not self.p1.pinned and not self.p2.pinned:
            self.p1.position += correction * 0.5
            self.p2.position -= correction * 0.5
        elif self.p1.pinned:
            self.p2.position -= correction
        elif self.p2.pinned:
            self.p1.position += correction


class Triangle:
    """Represents a triangle in the cloth mesh"""
    def __init__(self, p1, p2, p3):
        self.particles = [p1, p2, p3]
        self.active = True
    
    def get_normal(self):
        """Calculate triangle normal for rendering"""
        if not self.active:
            return np.array([0.0, 0.0, 1.0])
        
        v1 = self.particles[1].position - self.particles[0].position
        v2 = self.particles[2].position - self.particles[0].position
        normal = np.cross(v1, v2)
        length = np.linalg.norm(normal)
        return normal / length if length > 0.0001 else np.array([0.0, 0.0, 1.0])
    
    def intersects_line(self, line_start, line_end):
        """Check if line segment intersects this triangle"""
        if not self.active:
            return False, None
        
        v0 = self.particles[0].position
        v1 = self.particles[1].position
        v2 = self.particles[2].position
        
        # Check if line is close to triangle in Z-axis (within layer thickness)
        triangle_z = (v0[2] + v1[2] + v2[2]) / 3.0
        line_z = (line_start[2] + line_end[2]) / 2.0
        
        if abs(triangle_z - line_z) > 0.1:  # Too far in Z direction
            return False, None
        
        # Project to 2D (XY plane) for simpler intersection
        line_start_2d = line_start[:2]
        line_end_2d = line_end[:2]
        v0_2d = v0[:2]
        v1_2d = v1[:2]
        v2_2d = v2[:2]
        
        # Check if line segment intersects any triangle edge
        edges = [(v0_2d, v1_2d), (v1_2d, v2_2d), (v2_2d, v0_2d)]
        
        for edge_start, edge_end in edges:
            if self._segments_intersect_2d(line_start_2d, line_end_2d, edge_start, edge_end):
                # Calculate 3D intersection point (approximate)
                mid_point = (line_start + line_end) / 2.0
                return True, mid_point
        
        # Also check if line passes through triangle interior
        if self._point_in_triangle_2d(line_start_2d, v0_2d, v1_2d, v2_2d) or \
           self._point_in_triangle_2d(line_end_2d, v0_2d, v1_2d, v2_2d):
            mid_point = (line_start + line_end) / 2.0
            return True, mid_point
        
        return False, None
    
    def _segments_intersect_2d(self, p1, p2, p3, p4):
        """Check if two 2D line segments intersect"""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def _point_in_triangle_2d(self, p, v0, v1, v2):
        """Check if point is inside triangle (2D)"""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(p, v0, v1)
        d2 = sign(p, v1, v2)
        d3 = sign(p, v2, v0)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)


class ClothSimulation:
    """Multi-layer cloth simulation with cutting support"""
    def __init__(self, width=20, height=20, spacing=0.1, layers=2, layer_spacing=0.05):
        self.width = width
        self.height = height
        self.spacing = spacing
        self.layers = layers
        self.layer_spacing = layer_spacing
        
        self.particles = []
        self.constraints = []
        self.triangles = []
        
        self.gravity = np.array([0.0, -9.81, 0.0])
        self.damping = 0.98
        self.constraint_iterations = 5
        
        self.generate_cloth()
    
    def generate_cloth(self):
        """Generate multi-layer cloth mesh"""
        print(f"Generating cloth: {self.width}x{self.height}, {self.layers} layers...")
        
        # Create particles for each layer
        particle_grid = []
        for layer in range(self.layers):
            layer_particles = []
            z_offset = layer * self.layer_spacing
            
            for y in range(self.height + 1):
                row = []
                for x in range(self.width + 1):
                    px = (x - self.width / 2.0) * self.spacing
                    py = (self.height / 2.0 - y) * self.spacing
                    pz = z_offset
                    
                    particle = Particle(px, py, pz)
                    
                    # Pin top corners of first layer
                    if layer == 0 and y == 0 and (x == 0 or x == self.width):
                        particle.pinned = True
                    
                    self.particles.append(particle)
                    row.append(particle)
                layer_particles.append(row)
            particle_grid.append(layer_particles)
        
        # Create structural constraints within each layer
        for layer in range(self.layers):
            for y in range(self.height + 1):
                for x in range(self.width + 1):
                    p = particle_grid[layer][y][x]
                    
                    if x < self.width:
                        self.constraints.append(Constraint(p, particle_grid[layer][y][x + 1]))
                    if y < self.height:
                        self.constraints.append(Constraint(p, particle_grid[layer][y + 1][x]))
                    
                    # Diagonal constraints
                    if x < self.width and y < self.height:
                        self.constraints.append(Constraint(p, particle_grid[layer][y + 1][x + 1]))
                        self.constraints.append(Constraint(particle_grid[layer][y][x + 1], 
                                                          particle_grid[layer][y + 1][x]))
        
        # Connect layers
        if self.layers > 1:
            for layer in range(self.layers - 1):
                for y in range(self.height + 1):
                    for x in range(self.width + 1):
                        p1 = particle_grid[layer][y][x]
                        p2 = particle_grid[layer + 1][y][x]
                        self.constraints.append(Constraint(p1, p2, stiffness=0.99))
        
        # Create triangles for each layer
        for layer in range(self.layers):
            for y in range(self.height):
                for x in range(self.width):
                    p1 = particle_grid[layer][y][x]
                    p2 = particle_grid[layer][y][x + 1]
                    p3 = particle_grid[layer][y + 1][x]
                    p4 = particle_grid[layer][y + 1][x + 1]
                    
                    self.triangles.append(Triangle(p1, p2, p3))
                    self.triangles.append(Triangle(p2, p4, p3))
        
        print(f"Created {len(self.particles)} particles, {len(self.constraints)} constraints, {len(self.triangles)} triangles")
    
    def update(self, dt):
        """Update cloth physics simulation"""
        # Apply gravity to all particles
        for particle in self.particles:
            if not particle.pinned:
                particle.apply_force(self.gravity * particle.mass)
        
        # Update particle positions
        for particle in self.particles:
            particle.update(dt, self.damping)
        
        # Satisfy constraints multiple times for stability (more iterations = less stretch)
        for _ in range(self.constraint_iterations):
            for constraint in self.constraints:
                constraint.satisfy()
        
        # Prevent excessive stretching - additional constraint pass
        for constraint in self.constraints:
            if constraint.active:
                delta = constraint.p2.position - constraint.p1.position
                current_length = np.linalg.norm(delta)
                # Prevent stretching beyond 150% of rest length
                max_length = constraint.rest_length * 1.5
                if current_length > max_length:
                    correction = delta * (current_length - max_length) / current_length
                    if not constraint.p1.pinned and not constraint.p2.pinned:
                        constraint.p1.position += correction * 0.5
                        constraint.p2.position -= correction * 0.5
                    elif constraint.p1.pinned:
                        constraint.p2.position -= correction
                    elif constraint.p2.pinned:
                        constraint.p1.position += correction
    
    def cut_with_line(self, line_start, line_end):
        """Cut cloth along a line"""
        print(f"Cutting from {line_start} to {line_end}")
        
        cut_count = 0
        affected_particles = set()
        
        # Find intersecting triangles
        for triangle in self.triangles:
            intersects, point = triangle.intersects_line(line_start, line_end)
            if intersects:
                triangle.active = False
                cut_count += 1
                
                # Mark particles for potential cutting
                for p in triangle.particles:
                    affected_particles.add(p)
        
        # Only disable constraints that directly cross the cut line
        for constraint in self.constraints:
            if not constraint.active:
                continue
            
            # Check if constraint crosses the cut line
            if self._constraint_crosses_line(constraint, line_start, line_end):
                constraint.active = False
        
        print(f"Cut {cut_count} triangles, affected {len(affected_particles)} particles")
        return cut_count > 0
    
    def _constraint_crosses_line(self, constraint, line_start, line_end):
        """Check if a constraint (spring) crosses the cut line"""
        p1_pos = constraint.p1.position[:2]  # 2D position
        p2_pos = constraint.p2.position[:2]
        line_start_2d = line_start[:2]
        line_end_2d = line_end[:2]
        
        # Check if the constraint segment intersects the cut line segment
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        # Line segments intersect if they straddle each other
        return (ccw(p1_pos, line_start_2d, line_end_2d) != ccw(p2_pos, line_start_2d, line_end_2d) and
                ccw(p1_pos, p2_pos, line_start_2d) != ccw(p1_pos, p2_pos, line_end_2d))
    
    def _is_on_cut_side(self, point, line_start, line_end):
        """Determine if point is on the "removed" side of cut line"""
        # Simple 2D check (ignoring Z for now)
        line_dir = line_end[:2] - line_start[:2]
        perp = np.array([-line_dir[1], line_dir[0]])
        to_point = point[:2] - line_start[:2]
        return np.dot(to_point, perp) < 0


class ClothCuttingApp:
    """OpenGL application for cloth cutting visualization"""
    def __init__(self):
        self.cloth = None
        self.last_time = time.time()
        
        # Cutting state
        self.cut_start = None
        self.cut_end = None
        self.is_drawing_cut = False
        
        # Camera
        self.camera_rotation_x = 20
        self.camera_rotation_y = 0
        self.camera_distance = 5.0
        
        # Mouse
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_button = None
    
    def init_gl(self):
        """Initialize OpenGL settings"""
        glClearColor(0.2, 0.2, 0.3, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Setup lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 5.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        # Create cloth
        self.cloth = ClothSimulation(width=15, height=15, spacing=0.15, layers=2)
    
    def display(self):
        """Render the scene"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Setup camera
        gluLookAt(0, 0, self.camera_distance, 0, 0, 0, 0, 1, 0)
        glRotatef(self.camera_rotation_x, 1, 0, 0)
        glRotatef(self.camera_rotation_y, 0, 1, 0)
        
        # Draw cloth
        self.draw_cloth()
        
        # Draw cut line preview
        if self.cut_start is not None and self.cut_end is not None:
            self.draw_cut_line()
        
        glutSwapBuffers()
    
    def draw_cloth(self):
        """Render cloth triangles"""
        glColor3f(0.7, 0.8, 0.9)
        
        glBegin(GL_TRIANGLES)
        for triangle in self.cloth.triangles:
            if triangle.active:
                normal = triangle.get_normal()
                glNormal3fv(normal)
                
                for particle in triangle.particles:
                    glVertex3fv(particle.position)
        glEnd()
        
        # Draw wireframe
        glDisable(GL_LIGHTING)
        glColor3f(0.3, 0.3, 0.4)
        glLineWidth(1.0)
        
        glBegin(GL_LINES)
        for constraint in self.cloth.constraints:
            if constraint.active:
                glVertex3fv(constraint.p1.position)
                glVertex3fv(constraint.p2.position)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_cut_line(self):
        """Draw the cutting line preview"""
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 0.0, 0.0)
        glLineWidth(3.0)
        
        glBegin(GL_LINES)
        glVertex3fv(self.cut_start)
        glVertex3fv(self.cut_end)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def update(self):
        """Update physics and trigger redraw"""
        current_time = time.time()
        dt = min(current_time - self.last_time, 0.016)  # Cap at 60 FPS
        self.last_time = current_time
        
        self.cloth.update(dt)
        
        glutPostRedisplay()
    
    def reshape(self, width, height):
        """Handle window reshape"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
    
    def keyboard(self, key, x, y):
        """Handle keyboard input"""
        if key == b'q' or key == b'\x1b':  # Q or ESC
            glutLeaveMainLoop()
        elif key == b'c':  # Cut with random line
            self.perform_random_cut()
        elif key == b'r':  # Reset
            self.cloth = ClothSimulation(width=15, height=15, spacing=0.15, layers=2)
            print("Cloth reset")
        elif key == b' ':  # Space - cut horizontally
            self.perform_horizontal_cut()
    
    def mouse(self, button, state, x, y):
        """Handle mouse button events"""
        self.last_mouse_x = x
        self.last_mouse_y = y
        
        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                self.mouse_button = GLUT_LEFT_BUTTON
                # Start drawing cut line
                self.is_drawing_cut = True
                self.cut_start = self.screen_to_world(x, y)
                self.cut_end = self.cut_start.copy()
            else:
                self.mouse_button = None
                # Finish cut
                if self.is_drawing_cut and self.cut_start is not None:
                    self.cloth.cut_with_line(self.cut_start, self.cut_end)
                self.is_drawing_cut = False
        
        elif button == GLUT_RIGHT_BUTTON:
            if state == GLUT_DOWN:
                self.mouse_button = GLUT_RIGHT_BUTTON
            else:
                self.mouse_button = None
    
    def motion(self, x, y):
        """Handle mouse motion"""
        dx = x - self.last_mouse_x
        dy = y - self.last_mouse_y
        
        if self.mouse_button == GLUT_RIGHT_BUTTON:
            # Rotate camera
            self.camera_rotation_y += dx * 0.5
            self.camera_rotation_x += dy * 0.5
        elif self.is_drawing_cut:
            # Update cut line end
            self.cut_end = self.screen_to_world(x, y)
        
        self.last_mouse_x = x
        self.last_mouse_y = y
    
    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates (simplified)"""
        # Simple approximation - project onto cloth plane
        viewport = glGetIntegerv(GL_VIEWPORT)
        height = viewport[3]
        
        # Normalize coordinates
        nx = (screen_x / viewport[2]) * 2 - 1
        ny = 1 - (screen_y / height) * 2
        
        # Approximate world position on cloth plane
        world_x = nx * 2.0
        world_y = ny * 2.0
        world_z = 0.025  # At cloth center between layers
        
        return np.array([world_x, world_y, world_z])
    
    def perform_random_cut(self):
        """Perform a random cut"""
        start = np.array([np.random.uniform(-1.5, 1.5), 
                         np.random.uniform(-1.5, 1.5), 0.025])
        end = np.array([np.random.uniform(-1.5, 1.5), 
                       np.random.uniform(-1.5, 1.5), 0.025])
        self.cloth.cut_with_line(start, end)
        print("Random cut performed")
    
    def perform_horizontal_cut(self):
        """Perform a horizontal cut through the middle"""
        start = np.array([-2.0, 0.0, 0.025])
        end = np.array([2.0, 0.0, 0.025])
        self.cloth.cut_with_line(start, end)
        print("Horizontal cut performed")
    
    def run(self):
        """Start the application"""
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(800, 600)
        glutCreateWindow(b"Cloth Cutting Simulation")
        
        self.init_gl()
        
        glutDisplayFunc(self.display)
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.motion)
        glutIdleFunc(self.update)
        
        print("\n=== CONTROLS ===")
        print("Left Mouse: Click & drag to draw cut line")
        print("Right Mouse: Drag to rotate camera")
        print("SPACE: Horizontal cut")
        print("C: Random cut")
        print("R: Reset cloth")
        print("Q/ESC: Quit")
        print("================\n")
        
        glutMainLoop()


if __name__ == "__main__":
    app = ClothCuttingApp()
    app.run()

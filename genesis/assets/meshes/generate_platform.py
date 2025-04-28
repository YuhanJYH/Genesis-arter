import numpy as np
import trimesh

# --- Configuration ---
top_width = 10.0  # Width/Length of the top square platform (meters)
height = 5.0      # Height of the platform (meters)
slope_angle_deg = 45.0 # Angle of the slopes in degrees
output_filename = "terrain_platform.obj" # Name of the output file

# --- Calculations ---
# Calculate the horizontal extent of the slope
# For a 45-degree slope, horizontal extent = vertical height
if np.isclose(slope_angle_deg, 45.0):
    horizontal_extent = height
else:
    # General case (though the request specified 45)
    slope_angle_rad = np.radians(slope_angle_deg)
    # Ensure tan(slope_angle_rad) is not zero to avoid division by zero
    if np.isclose(np.tan(slope_angle_rad), 0.0):
        raise ValueError("Slope angle cannot be 0 or 180 degrees.")
    horizontal_extent = height / np.tan(slope_angle_rad)

# Calculate base dimensions
base_width = top_width + 2 * horizontal_extent

# Calculate half-widths for easier coordinate definition (assuming centered at origin)
half_top = top_width / 2.0
half_base = base_width / 2.0

# --- Define Vertices ---
# 8 vertices: 4 for the base, 4 for the top platform
# Vertices are defined as [x, y, z]
vertices = np.array([
    # Base vertices (z=0) - Indices 0 to 3
    [-half_base, -half_base, 0.0],  # 0: Front-Left-Bottom
    [ half_base, -half_base, 0.0],  # 1: Front-Right-Bottom
    [ half_base,  half_base, 0.0],  # 2: Back-Right-Bottom
    [-half_base,  half_base, 0.0],  # 3: Back-Left-Bottom

    # Top vertices (z=height) - Indices 4 to 7
    [-half_top, -half_top, height], # 4: Front-Left-Top
    [ half_top, -half_top, height], # 5: Front-Right-Top
    [ half_top,  half_top, height], # 6: Back-Right-Top
    [-half_top,  half_top, height]  # 7: Back-Left-Top
])

# --- Define Faces ---
# Define the faces as triangles using vertex indices.
# 6 faces (top, bottom, 4 sides), each split into 2 triangles = 12 triangles total.
# Winding order (counter-clockwise when viewed from outside) matters for normals.
faces = np.array([
    # Bottom face (z=0) - Triangles using vertices 0, 1, 2, 3
    # Normal should point down (-Z)
    [0, 1, 2], # Triangle 1
    [0, 2, 3], # Triangle 2

    # Top face (z=height) - Triangles using vertices 4, 5, 6, 7
    # *** CORRECTED WINDING ORDER *** Normal should point up (+Z)
    [4, 5, 6], # Triangle 1 (Previously [4, 6, 5])
    [4, 6, 7], # Triangle 2 (Previously [4, 7, 6])

    # Side faces - Each side is a quad split into two triangles
    # Normals should point outwards from the center.
    # Front face (-Y direction) - Vertices 0, 1, 5, 4
    [0, 1, 5],
    [0, 5, 4],

    # Right face (+X direction) - Vertices 1, 2, 6, 5
    [1, 2, 6],
    [1, 6, 5],

    # Back face (+Y direction) - Vertices 2, 3, 7, 6
    [2, 3, 7],
    [2, 7, 6],

    # Left face (-X direction) - Vertices 3, 0, 4, 7
    [3, 0, 4],
    [3, 4, 7]
])

# --- Create the Mesh ---
# Create a trimesh.Trimesh object from the vertices and faces
# process=False prevents trimesh from potentially reordering faces/vertices
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

# --- Verify and Export ---
# Optional: Check if the mesh is watertight and print basic info
# Manually check normals for the top face if needed
mesh.fix_normals() # Ensure normals are consistent after potential manual definition issues
print(f"Mesh is watertight: {mesh.is_watertight}")
print(f"Mesh is convex: {mesh.is_convex}")
print(f"Mesh volume: {mesh.volume}") # Should be > 0

# Export the mesh to an OBJ file
try:
    # Export with options to ensure vertex order isn't changed if possible
    mesh.export(output_filename, include_normals=True)
    print(f"Successfully exported mesh to '{output_filename}'")
except Exception as e:
    print(f"Error exporting mesh: {e}")


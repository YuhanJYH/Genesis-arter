import numpy as np
import trimesh


def create_sloped_platform(top_width: float, height: float, slope_angle_deg: float, output_path: str):
    """
    Generates a 3D mesh of a platform with sloped sides and exports it as an STL file.

    Args:
        top_width (float): Width/Length of the top square platform (in meters).
        height (float): Height of the platform (in meters).
        slope_angle_deg (float): Angle of the slopes in degrees.
        output_path (str): Path to save the generated STL file.
    """
    # Calculate the horizontal extent of the slope
    if np.isclose(slope_angle_deg, 45.0):
        horizontal_extent = height
    else:
        slope_angle_rad = np.radians(slope_angle_deg)
        if np.isclose(np.tan(slope_angle_rad), 0.0):
            raise ValueError("Slope angle cannot be 0 or 180 degrees.")
        horizontal_extent = height / np.tan(slope_angle_rad)

    # Calculate base dimensions
    base_width = top_width + 2 * horizontal_extent

    # Calculate half-widths for easier coordinate definition (assuming centered at origin)
    half_top = top_width / 2.0
    half_base = base_width / 2.0

    # Define Vertices
    vertices = np.array(
        [
            # Base vertices (z=0)
            [-half_base, -half_base, 0.0],  # 0: Front-Left-Bottom
            [half_base, -half_base, 0.0],  # 1: Front-Right-Bottom
            [half_base, half_base, 0.0],  # 2: Back-Right-Bottom
            [-half_base, half_base, 0.0],  # 3: Back-Left-Bottom
            # Top vertices (z=height)
            [-half_top, -half_top, height],  # 4: Front-Left-Top
            [half_top, -half_top, height],  # 5: Front-Right-Top
            [half_top, half_top, height],  # 6: Back-Right-Top
            [-half_top, half_top, height],  # 7: Back-Left-Top
        ]
    )

    # Define Faces
    faces = np.array(
        [
            # Bottom face (z=0)
            [0, 1, 2],
            [0, 2, 3],
            # Top face (z=height)
            [4, 5, 6],
            [4, 6, 7],
            # Side faces
            # Front face (-Y direction)
            [0, 1, 5],
            [0, 5, 4],
            # Right face (+X direction)
            [1, 2, 6],
            [1, 6, 5],
            # Back face (+Y direction)
            [2, 3, 7],
            [2, 7, 6],
            # Left face (-X direction)
            [3, 0, 4],
            [3, 4, 7],
        ]
    )

    platform_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    platform_mesh.fix_normals()
    platform_mesh.remove_duplicate_faces()
    platform_mesh.remove_unreferenced_vertices()
    platform_mesh.remove_infinite_values()
    platform_mesh.export(file_obj=output_path, file_type="stl")
    print(f"stl file saved to: {output_path}")


if __name__ == "__main__":
    # --- Configuration ---
    top_width = 10.0  # Width/Length of the top square platform (meters)
    height = 5.0  # Height of the platform (meters)
    slope_angle_deg = 80.0  # Angle of the slopes in degrees
    output_file = "/workspace/genesis/assets/meshes/platform.stl"

    create_sloped_platform(top_width, height, slope_angle_deg, output_file)
    print(f"STL file saved to: {output_file}")

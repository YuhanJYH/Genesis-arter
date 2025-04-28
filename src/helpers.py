def create_constrained_random_heightfield(size_meters, max_height_diff, points_per_meter=10):
    """
    Generates a random height field with a specified size and maximum height difference.

    Args:
        size_meters (float): The physical size of the terrain in meters (both width and length).
        max_height_diff (float): The maximum allowable difference in height within the terrain.
        points_per_meter (int, optional): The number of data points (grid cells) per meter.
            Higher values create a more detailed height field. Defaults to 10.

    Returns:
        numpy.ndarray: A 2D NumPy array representing the height field.
            The dimensions of the array are (size_meters * points_per_meter) x (size_meters * points_per_meter).
    """
    # Calculate the number of points (resolution) based on points_per_meter
    num_points = int(size_meters * points_per_meter)

    # Create a base height field with small random values.  We use float32
    # because it's common in 3D graphics and terrain generation.
    height_field = np.random.rand(num_points, num_points).astype(np.float32) * (max_height_diff / 10) # Start with a fraction of max_height_diff

    # Apply a filter to smooth the height field and enforce the maximum height difference.
    # We'll use a simple approach here:
    for _ in range(5):  # Iterative smoothing (more iterations = smoother terrain)
        height_field = (height_field[1:-1, 1:-1] +
                        height_field[0:-2, 1:-1] +
                        height_field[2:, 1:-1] +
                        height_field[1:-1, 0:-2] +
                        height_field[1:-1, 2:] +
                        height_field[0:-2, 0:-2] +
                        height_field[2:, 0:-2] +
                        height_field[0:-2, 2:] +
                        height_field[2:, 2:]) / 9.0
        height_field = np.clip(height_field, 0, max_height_diff) # Clamp to the max height difference

    return height_field
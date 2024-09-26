import jax.numpy as jnp

def generate_filter(num_channels, filter="Ram-Lak"):
    """
    Creates the specified filter in the time domain of size (2*num_channels - 1).

    Currently supported filters include: \"Ram-Lak\", \"Shepp-Logan.\"

    Args:
        num_channels (int): Number of detector channels in the sinogram.
        filter (string, optional): Name of the filter to be generated. Defaults to "Ram-Lak."

    Returns:
        filter (jnp): The computed filter (filter.size = 2*num_channels + 1).
    """
    # If you want to add a new filter, place it's name into supported_filters, and ...
    # ... create a new if statement with the filter math.
    supported_filters = ["Ram-Lak", "Shepp-Logan"]
    n = jnp.arange(-num_channels + 1, num_channels)  # ex: num_channels = 3, -> n = [-2, -1, 0, 1, 2]
    # Raise error if inputed filter is not supported.
    if filter not in supported_filters:
        raise ValueError(f"Unsupported filter. Supported filters are: {', '.join(supported_filters)}.")
    
    if filter == "Ram-Lak":
        filter = (1 / 2) * jnp.sinc(n) - (1 / 4) * (jnp.sinc(n / 2)) ** 2

    if filter == "Shepp-Logan":
        filter = (-2) / ((jnp.pi ** 2) * (4 * (n)**2 - 1))

    return filter



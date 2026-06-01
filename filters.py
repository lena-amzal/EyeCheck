import numpy as np

def moving_average(data, window_size=5):
    """Calcule la moyenne glissante pour lisser le signal de pression."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')
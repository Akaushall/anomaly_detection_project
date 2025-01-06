# Import necessary libraries
import numpy as np
import random
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Step 1 & 2: Data Stream Simulation Function
def data_stream_simulation(size=1000, seasonality=50, noise_level=5):
    for i in range(size):
        seasonal_component = 10 * np.sin(2 * np.pi * i / seasonality)
        random_noise = random.gauss(0, noise_level)
        value = 50 + seasonal_component + random_noise
        yield value
        time.sleep(0.1)

# Step 3: Anomaly Detection Class
class AnomalyDetector:
    def __init__(self, window_size=30, threshold=3):
        self.window_size = window_size
        self.threshold = threshold
        self.window = deque(maxlen=window_size)

    def detect_anomaly(self, value):
        self.window.append(value)
        if len(self.window) < self.window_size:
            return False
        mean = np.mean(self.window)
        std_dev = np.std(self.window)
        z_score = (value - mean) / std_dev if std_dev != 0 else 0
        return abs(z_score) > self.threshold

# Step 5: Visualization Function
def visualize_data_stream(detector, stream_generator, size=100):
    data = []
    anomalies = []
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    scatter = ax.scatter([], [], color='red', label="Anomaly")
    ax.set_ylim(30, 70)
    ax.set_xlim(0, size)

    def init():
        line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        return line, scatter

    def update(frame):
        value = next(stream_generator)
        is_anomaly = detector.detect_anomaly(value)
        data.append(value)
        if is_anomaly:
            anomalies.append((len(data) - 1, value))
        line.set_data(range(len(data)), data)
        scatter.set_offsets(np.array(anomalies))
        ax.set_xlim(max(0, len(data) - size), len(data))
        return line, scatter

    ani = FuncAnimation(fig, update, frames=size, init_func=init, blit=True, repeat=False)
    plt.legend()
    plt.show()

# Execute the Anomaly Detection Project
detector = AnomalyDetector(window_size=30, threshold=3)
stream = data_stream_simulation()
visualize_data_stream(detector, stream)

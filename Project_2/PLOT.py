import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_flight_log(csv_path='flight_log.csv'):
    """
    读取 flight_log.csv 并绘制三维轨迹
    """
    df = pd.read_csv(csv_path)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df['x'], df['y'], df['z'], label='Flight Path', color='blue')
    ax.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], color='green', label='Start')
    ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], color='red', label='End')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Flight Trajectory')
    ax.legend()
    plt.show()
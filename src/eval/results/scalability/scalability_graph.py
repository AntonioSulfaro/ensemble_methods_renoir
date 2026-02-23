import matplotlib.pyplot as plt
import pandas as pd

# 1. Load your CSV file
file_path = 'threads_time'
df = pd.read_csv(file_path + '.csv')

# 3. Create the plot
plt.figure(figsize=(12, 6))

plt.plot(df['n_threads'],
         df['time'],
         marker='o',  # Add points
         linestyle='-',  # Connect with lines
         color='#2c7bb6',  # Nice blue color
         markersize=3,  # Size of the dots
         linewidth=1,  # Width of the line
         alpha=0.8)  # Transparency

# 4. Formatting the chart
plt.title('Global time execution', fontsize=14, fontweight='bold')
plt.xlabel('Thread number', fontsize=12)
plt.ylabel('Time [s]', fontsize=12)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Tight layout to prevent labels from being cut off
plt.tight_layout()

# 6. Show or Save the plot
plt.savefig(file_path + '_time.png', dpi=300)
# plt.show()

plt.figure(figsize=(12, 6))

plt.plot(df['n_threads'],
         1e6 / df['time'],
         marker='o',  # Add points
         linestyle='-',  # Connect with lines
         color='#2c7bb6',  # Nice blue color
         markersize=3,  # Size of the dots
         linewidth=1,  # Width of the line
         alpha=0.8)  # Transparency

# 4. Formatting the chart
plt.title('Final throughput', fontsize=14, fontweight='bold')
plt.xlabel('Thread number', fontsize=12)
plt.ylabel('Throughput [instances/s]', fontsize=12)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Tight layout to prevent labels from being cut off
plt.tight_layout()

# 6. Show or Save the plot
plt.savefig(file_path + '_throughput.png', dpi=300)
# plt.show()

import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv('time_to_train_data.csv')

# Create a line plot using matplotlib to visualize the loss of the quantum and classical models over the 10 iterations
plt.plot(df['#'], df['Quantum'], label='Quantum')
plt.plot(df['#'], df['Classical'], label='Classical')
plt.xlabel('Iteration')
plt.ylabel('Time (seconds)')
plt.title('Time Comparison')
plt.legend()

# Create a folder named 'time' if it doesn't exist
if not os.path.exists('time'):
    os.makedirs('time')

# Save the plot as a PNG file in the 'time' folder
plt.savefig('time/time_train_plot.png')

# Show the plot
plt.show()

# Calculate the mean, median, and standard deviation of the time for both the quantum and classical models
quantum_mean = df['Quantum'].mean()
quantum_median = df['Quantum'].median()
quantum_std = df['Quantum'].std()

classical_mean = df['Classical'].mean()
classical_median = df['Classical'].median()
classical_std = df['Classical'].std()

# Print out the statistical data for both models
print('Quantum Model:')
print('Mean Time:', quantum_mean)
print('Median Time:', quantum_median)
print('Standard Deviation:', quantum_std)

print('\nClassical Model:')
print('Mean Time:', classical_mean)
print('Median Time:', classical_median)
print('Standard Deviation:', classical_std)


# Save the statistical data for both models to a CSV file in the 'time' folder
stats_df = pd.DataFrame({
    'Model': ['Quantum', 'Classical'],
    'Mean Time (s)': [quantum_mean, classical_mean],
    'Median Time (s)': [quantum_median, classical_median],
    'Standard Deviation (s)': [quantum_std, classical_std]
})
stats_df.to_csv('time/time_train_stats.csv', index=False)

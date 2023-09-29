import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv('accuracy_data.csv')

# Create a line plot using matplotlib to visualize the accuracy of the quantum and classical models over the 10 iterations
plt.plot(df['#'], df['Quantum'], label='Quantum')
plt.plot(df['#'], df['Classical'], label='Classical')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend()

# Create a folder named 'accuracy' if it doesn't exist
if not os.path.exists('accuracy'):
    os.makedirs('accuracy')

# Save the plot as a PNG file in the 'accuracy' folder
plt.savefig('accuracy/accuracy_plot.png')

# Show the plot
plt.show()

# Calculate the mean, median, and standard deviation of the accuracy for both the quantum and classical models
quantum_mean = df['Quantum'].mean()
quantum_median = df['Quantum'].median()
quantum_std = df['Quantum'].std()

classical_mean = df['Classical'].mean()
classical_median = df['Classical'].median()
classical_std = df['Classical'].std()

# Print out the statistical data for both models
print('Quantum Model:')
print('Mean Accuracy:', quantum_mean)
print('Median Accuracy:', quantum_median)
print('Standard Deviation:', quantum_std)

print('\nClassical Model:')
print('Mean Accuracy:', classical_mean)
print('Median Accuracy:', classical_median)
print('Standard Deviation:', classical_std)


# Save the statistical data for both models to a CSV file in the 'accuracy' folder
stats_df = pd.DataFrame({
    'Model': ['Quantum', 'Classical'],
    'Mean Accuracy': [quantum_mean, classical_mean],
    'Median Accuracy': [quantum_median, classical_median],
    'Standard Deviation': [quantum_std, classical_std]
})
stats_df.to_csv('accuracy/accuracy_stats.csv', index=False)

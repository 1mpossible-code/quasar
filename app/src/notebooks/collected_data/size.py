import pandas as pd

# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv('size_data.csv')

# Convert size from bytes to megabytes
df['Quantum'] = df['Quantum'] / (1024 * 1024)
df['Classical'] = df['Classical'] / (1024 * 1024)

# Calculate the mean, median, and standard deviation of the size for both the quantum and classical models
quantum_mean = df['Quantum'].mean()
quantum_median = df['Quantum'].median()
quantum_std = df['Quantum'].std()

classical_mean = df['Classical'].mean()
classical_median = df['Classical'].median()
classical_std = df['Classical'].std()

# Print out the statistical data for both models
print('Quantum Model:')
print('Mean Size (MB):', quantum_mean)
print('Median Size (MB):', quantum_median)
print('Standard Deviation (MB):', quantum_std)

print('\nClassical Model:')
print('Mean Size (MB):', classical_mean)
print('Median Size (MB):', classical_median)
print('Standard Deviation (MB):', classical_std)

# Compare the statistical data for both models
print('\nComparison:')
print('Mean Size Difference (MB):', abs(quantum_mean - classical_mean))
print('Median Size Difference (MB):', abs(quantum_median - classical_median))
print('Standard Deviation Difference (MB):', abs(quantum_std - classical_std))



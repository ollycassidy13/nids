import glob
import pandas as pd

csv_files = glob.glob('dataset/MachineLearningCVE/*.csv')
print(f"Found CSV files: {csv_files}")

# Load a single CSV file to inspect its columns
df = pd.read_csv('dataset/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
print(df.columns)


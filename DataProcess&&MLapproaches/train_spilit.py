import pandas as pd

# Function to split a CSV file into smaller CSV files
def split_csv(input_file, output_prefix, chunk_size):
    # Read the large CSV file in chunks
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        # Define the output file name
        output_file = f'{output_prefix}{i+1}.csv'
        # Write the chunk to a new CSV file
        chunk.to_csv(output_file, index=False)
        print(f'Written {len(chunk)} rows to {output_file}')

# Specify the input CSV file, output prefix, and chunk size
input_file = 'train.csv'
output_prefix = 'train'
chunk_size = 10000

# Call the function to split the CSV file
split_csv(input_file, output_prefix, chunk_size)

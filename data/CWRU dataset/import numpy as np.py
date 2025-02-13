import numpy as np
import pandas as pd

def npz_to_csv(npz_filepath, csv_filepath):
    try:
        data = np.load(npz_filepath)
        arrays = [data[key] for key in data.files]

        if len(arrays) == 1:
            df = pd.DataFrame(arrays[0].reshape(arrays[0].shape[0], -1)) # Flatten if 3D
        else:
            # Flatten the first array if it's 3D
            if len(arrays[0].shape) > 2:
                arrays[0] = arrays[0].reshape(arrays[0].shape[0], -1)
            df = pd.concat([pd.DataFrame(arr) for arr in arrays], axis=1)

        df.to_csv(csv_filepath, index=False)
        print(f"Data saved to {csv_filepath}")

    except FileNotFoundError:
        print(f"Error: NPZ file not found at {npz_filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
npz_file = r'C:\Users\SEMEH SAYADI\OneDrive - SHERPA Engineering\Bureau\Bearing_Fault Detection\data\CWRU dataset\CWRU_48k_load_1_CNN_data.npz'  
csv_file = 'CWRU_data.csv'
npz_to_csv(npz_file, csv_file)


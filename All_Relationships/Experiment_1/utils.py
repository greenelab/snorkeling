import numpy as np
import tqdm


def read_csv_file(filename, get_max_val=False):
    """
    Read a csv file into a numpy array for the LSTM.
    keywords:
    filename - name of the file to be read
    get_max_val- returns the maximum value of the total file <- used by the LSTM object
    """
    data = []
    val = 0
    with open(filename, 'rb') as f:
        input_file = csv.reader(f)
        for row in tqdm.tqdm(input_file):
            data.append(np.array(map(int, row)))
            if get_max_val:
                max_row = max(map(int, row))
                val = max_row if max_row > val else val
    return (np.array(data), (val+1)) if get_max_val else np.array(data)


def read_word_dict(filename):
    """
     Read a CSV into a dictionary using the Key column (as string) and Value column (as int).
    Keywords:
    fielname - name of the file to read
    """
    data = {}
    with open(filename, 'rb') as f:
        input_file = csv.DictReader(f)
        for row in tqdm.tqdm(input_file):
            data[row['Key']] = int(row['Value'])
    return data

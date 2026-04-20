import pickle

def load_data_from_pkl(file_path):
    """
    Load data form pickle file
    
    Args:
        file_path (str): The location of file.
    
    Returns:
        dictionary: the data loaded from the file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def save_population(filename, population):
    """
    Save the population data 
    
    Args:
        filename (str): The file location where we need to save the data.
        population (dict): The data to save
        
    Returns:
        None
    """
    with open(filename, 'wb') as fp:
        pickle.dump(population, fp)

import torch
from torch.utils.data import Dataset
from dataset.utils import load_pickle_data
from ripdalib.variations.transform import list_to_matrix, _get_mindiv


class FuguesDataset(Dataset):
    """
    Data Augmentation with flip pitches.
    NOTE : implement pitch transposition and time translation when possible.
    
    """

    def __init__(self, data_file:str, maxpitchdif:int=59, maxlength:float=481.0, mindiv:int=16):
        unprocessed = load_pickle_data(data_file)
        self.mindiv = mindiv
        self.names = list()
        self.data = list()

        for name, value in unprocessed.items():
            minpitch = min([note[1] for note in value])
            mintime = min([note[0] for note in value])
            translated = [(note[0] - mintime, note[1] - minpitch) for note in value]
            track = [note for note in translated if note[0] <= maxlength and note[1] <= maxpitchdif]

            matrix = torch.zeros((1, int((maxlength+1)*self.mindiv), maxpitchdif+1), dtype=torch.int8)
            for point in track:
                matrix[0, round((point[0])*self.mindiv), point[1]] = 1
            
            self.names.append(name)
            self.data.append(matrix)

        self.data = torch.stack(self.data)

    def __getitem__(self, index):
        return torch.concat((self.data, torch.flip(self.data, (-1,))))[index]
    
    def __len__(self):
        return self.data.shape[0] * 2

        
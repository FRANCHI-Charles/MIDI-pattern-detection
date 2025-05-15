from dataset.utils import load_pop909, save_dict

FOLDER_PATH = "/home/deck/Documents/Intership 2025/POP909-Dataset/POP909"

data = load_pop909(FOLDER_PATH, 12)
save_dict(data, "./pop909.pkl")
print("Success !")
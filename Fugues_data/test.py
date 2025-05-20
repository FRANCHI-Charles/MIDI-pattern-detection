import matplotlib.pyplot as plt

from utils import load_pickle_data

DATA_PATH = "processed/BWV_0859.pkl"
data = load_pickle_data(DATA_PATH)

print(data.keys())
try:
    plt.scatter(*zip(*data['Track 1'])) # 'Klavier rechte Hand' 'unbenannt'
except:
    pass

try:
    plt.scatter(*zip(*data['Track 2'])) # 'Klavier rechte Hand' 'unbenannt'
except:
    pass

try:
    plt.scatter(*zip(*data['Track 3'])) # 'Klavier rechte Hand' 'unbenannt'
except:
    pass

try:
    plt.scatter(*zip(*data['Track 4'])) # 'Klavier rechte Hand' 'unbenannt'
except:
    pass

plt.show()
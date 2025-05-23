import matplotlib.pyplot as plt

from utils import load_pickle_data


def check_shorten():
    DATA_PATH = "processed/BWV_0872.pkl"
    LENGTH = 200

    data = load_pickle_data(DATA_PATH)

    print(data.keys())
    try:
        plt.scatter(*zip(*data['Track 1'][:LENGTH])) # 'Klavier rechte Hand' 'unbenannt'
    except:
        pass

    try:
        plt.scatter(*zip(*data['Track 2'][:LENGTH])) # 'Klavier rechte Hand' 'unbenannt'
    except:
        pass

    try:
        plt.scatter(*zip(*data['Track 3'][:LENGTH])) # 'Klavier rechte Hand' 'unbenannt'
    except:
        pass

    try:
        plt.scatter(*zip(*data['Track 4'][:LENGTH])) # 'Klavier rechte Hand' 'unbenannt'
    except:
        pass

    plt.show()


def check_data():
    data = load_pickle_data("data.pkl")
    plt.scatter(*zip(*data["BWV_0859"]))
    plt.show()

check_data()
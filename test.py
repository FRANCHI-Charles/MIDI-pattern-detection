from dataset.utils import load_pickle_data
import matplotlib.pyplot as plt
import music21

def show_poppickle():
    data = load_pickle_data("./pop909.pkl")

    testtrack = data["001"]["PIANO"]
    print(data["001"].keys())

    print(testtrack[0])

    onsets = [elem[0] for elem in testtrack]
    pitches = [elem[1] for elem in testtrack]

    for note in onsets:
        if note.denominator > 24:
            print(note)

    plt.scatter(onsets, pitches)
    plt.show()

def trymusic21():
    score = music21.schema("05-bwv850.dez")
    print(score)

trymusic21()
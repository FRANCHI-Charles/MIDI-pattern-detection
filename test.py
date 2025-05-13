from dataset.utils import load_pickle_data
import matplotlib.pyplot as plt

data = load_pickle_data("./pop909.pkl")

testtrack = data["005"]["PIANO"]

print(testtrack[0])

onsets = [elem[0] for elem in testtrack]
pitches = [elem[1] for elem in testtrack]

plt.scatter(onsets, pitches)
plt.show()
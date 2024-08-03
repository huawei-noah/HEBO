import pickle
import sys

path = sys.argv[1]
with open(path, "rb") as f:
    data = pickle.load(f)


for data_point in data:
    print("=======================")
    print("Time:", data_point[0])
    for d in data_point[1:]:
        print("-----------------------")
        print(d)

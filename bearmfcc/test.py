import numpy as np
import wavio
from bearmfcc import process

if __name__ == "__main__":
    wavlist = list(wavio.read("./yes5333.wav").data.astype(float).flatten())
    features = process(wavlist,5333,16,96,96,12,50,2666)
    for item in features:
        print(item)
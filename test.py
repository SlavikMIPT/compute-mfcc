import numpy as np
import wavio
from bearmfcc import process

if __name__ == "__main__":
    wavlist = list(wavio.read("./yes5333.wav").data.astype(float).flatten())
    features = process(inList=wavlist, sampFreq=5333, nCep=16, winLength=96, frameShift=96, numFilt=12, lf=50, hf=2666)
    for item in features:
        print(item)
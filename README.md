# A Simple MFCC Feature Extractor using C++ STL and C++11

## Python module install

pip3 install .
python3 test.py

## Compilation

g++ -std=c++11 -O3 compute-mfcc.cc -o compute-mfcc

## Usage Examples

- compute-mfcc --input input.wav --output output.mfc
- compute-mfcc --input input.wav --output output.mfc --samplingrate 8000
- compute-mfcc --inputlist input.list --outputlist output.list
- compute-mfcc --inputlist input.list --outputlist output.list --numcepstra 17 --samplingrate 44100

## License

GNU GPL V3.0

## Contributors

D S Pavan Kumar

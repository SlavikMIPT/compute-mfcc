// -----------------------------------------------------------------------------
//  A simple MFCC extractor using C++ STL and C++11
// -----------------------------------------------------------------------------
//
//  Copyright (C) 2016 D S Pavan Kumar
//  dspavankumar [at] gmail [dot] com
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#ifndef __MFCC_H
#define __MFCC_H
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <map>
#include <numeric>
#include <vector>

#include "wavHeader.h"
// #include "ETL/vector.h"

using namespace std;
namespace Math {
typedef std::vector<float> v_d_t;
typedef std::complex<float> c_d_t;
typedef std::vector<v_d_t> m_d_t;
typedef std::vector<c_d_t> v_c_d_t;
typedef std::map<int, std::map<int, c_d_t>> twmap;

class MFCC {
 private:
  const float PI = 4 * atan(1.0);  // Pi = 3.14...
  int fs;
  twmap twiddle;
  size_t winLengthSamples, frameShiftSamples, numCepstra, numFFT, numFFTBins,
      numFilters;
  float preEmphCoef, lowFreq, highFreq;
  v_d_t frame, powerSpectralCoef, lmfbCoef, hamming, mfcc, prevsamples;
  m_d_t fbank, dct;

 private:
  // Hertz to Mel conversion
  inline float hz2mel(float f) { return 2595 * std::log10(1 + f / 700); }

  // Mel to Hertz conversion
  inline float mel2hz(float m) { return 700 * (std::pow(10, m / 2595) - 1); }

  // Twiddle factor computation
  void compTwiddle(void) {
    constexpr c_d_t J(0, 1);  // Imaginary number 'j'
    for (int N = 2; N <= numFFT; N *= 2)
      for (int k = 0; k <= N / 2 - 1; k++)
        twiddle[N][k] = exp(-2 * PI * k / N * J);
  }
  v_c_d_t fft(v_c_d_t x) {
    // DFT
    unsigned int N = x.size(), k = N, n;
    float thetaT = 3.14159265358979323846264338328L / N;
    c_d_t phiT = c_d_t(cos(thetaT), -sin(thetaT)), T;
    while (k > 1) {
      n = k;
      k >>= 1;
      phiT = phiT * phiT;
      T = 1.0L;
      for (unsigned int l = 0; l < k; l++) {
        for (unsigned int a = l; a < N; a += n) {
          unsigned int b = a + k;
          c_d_t t = x[a] - x[b];
          x[a] += x[b];
          x[b] = t * T;
        }
        T *= phiT;
      }
    }
    // Decimate
    unsigned int m = static_cast<unsigned int>(log2(N));
    for (unsigned int a = 0; a < N; a++) {
      unsigned int b = a;
      // Reverse bits
      b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
      b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
      b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
      b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
      b = ((b >> 16) | (b << 16)) >> (32 - m);
      if (b > a) {
        c_d_t t = x[a];
        x[a] = x[b];
        x[b] = t;
      }
    }
    return x;
    //// Normalize (This section make it not working correctly)
    // Complex f = 1.0 / sqrt(N);
    // for (unsigned int i = 0; i < N; i++)
    //	x[i] *= f;
  }
  //// Frame processing routines
  // Pre-emphasis and Hamming window
  void preEmphHam(void) {
    v_d_t procFrame(frame.size(), hamming[0] * frame[0]);
    for (int i = 1; i < frame.size(); i++)
      procFrame[i] = hamming[i] * (frame[i] - preEmphCoef * frame[i - 1]);
    frame = procFrame;
  }

  // Power spectrum computation
  void computePowerSpec(void) {
    frame.resize(numFFT);                          // Pads zeros
    v_c_d_t framec(frame.cbegin(), frame.cend());  // Complex frame
    v_c_d_t fftc = fft(framec);

    for (int i = 0; i < numFFTBins; i++)
      powerSpectralCoef[i] = pow(abs(fftc[i]), 2);
  }

  // Applying log Mel filterbank (LMFB)
  void applyLMFB(void) {
    lmfbCoef.assign(numFilters, 0);

    for (int i = 0; i < numFilters; i++) {
      // Multiply the filterbank matrix
      for (int j = 0; j < fbank[i].size(); j++)
        lmfbCoef[i] += fbank[i][j] * powerSpectralCoef[j];
      // Apply Mel-flooring
      if (lmfbCoef[i] < 1.0) lmfbCoef[i] = 1.0;
    }

    // Applying log on amplitude
    for (int i = 0; i < numFilters; i++) lmfbCoef[i] = std::log(lmfbCoef[i]);
  }

  // Computing discrete cosine transform
  void applyDct(void) {
    mfcc.assign(numCepstra, 0);
    for (int i = 0; i < numCepstra; i++) {
      for (int j = 0; j < numFilters; j++) mfcc[i] += dct[i][j] * lmfbCoef[j];
    }
  }

  // Initialisation routines
  // Pre-computing Hamming window and dct matrix
  void initHamDct(void) {
    int i, j;

    hamming.assign(winLengthSamples, 0);
    for (i = 0; i < winLengthSamples; i++)
      hamming[i] = 0.54 - 0.46 * cos(2 * PI * i / (winLengthSamples - 1));

    v_d_t v1(numCepstra, 0), v2(numFilters, 0);
    for (i = 0; i < numCepstra; i++) v1[i] = i;
    for (i = 0; i < numFilters; i++) v2[i] = i + 0.5;

    dct.reserve(numFilters * (numCepstra));
    float c = sqrt(2.0 / numFilters);
    for (i = 0; i < numCepstra; i++) {
      v_d_t dtemp;
      for (j = 0; j < numFilters; j++)
        dtemp.push_back(c * cos(PI / numFilters * v1[i] * v2[j]));
      dct.push_back(dtemp);
    }
  }

  // Precompute filterbank
  void initFilterbank() {
    // Convert low and high frequencies to Mel scale
    float lowFreqMel = hz2mel(lowFreq);
    float highFreqMel = hz2mel(highFreq);

    // Calculate filter centre-frequencies
    v_d_t filterCentreFreq;
    filterCentreFreq.reserve(numFilters + 2);
    for (int i = 0; i < numFilters + 2; i++)
      filterCentreFreq.push_back(mel2hz(
          lowFreqMel + (highFreqMel - lowFreqMel) / (numFilters + 1) * i));

    // Calculate FFT bin frequencies
    v_d_t fftBinFreq;
    fftBinFreq.reserve(numFFTBins);
    for (int i = 0; i < numFFTBins; i++)
      fftBinFreq.push_back(fs / 2.0 / (numFFTBins - 1) * i);

    // Filterbank: Allocate memory
    fbank.reserve(numFilters * numFFTBins);

    // Populate the fbank matrix
    for (int filt = 1; filt <= numFilters; filt++) {
      v_d_t ftemp;
      for (int bin = 0; bin < numFFTBins; bin++) {
        float weight;
        if (fftBinFreq[bin] < filterCentreFreq[filt - 1])
          weight = 0;
        else if (fftBinFreq[bin] <= filterCentreFreq[filt])
          weight = (fftBinFreq[bin] - filterCentreFreq[filt - 1]) /
                   (filterCentreFreq[filt] - filterCentreFreq[filt - 1]);
        else if (fftBinFreq[bin] <= filterCentreFreq[filt + 1])
          weight = (filterCentreFreq[filt + 1] - fftBinFreq[bin]) /
                   (filterCentreFreq[filt + 1] - filterCentreFreq[filt]);
        else
          weight = 0;
        ftemp.push_back(weight);
      }
      fbank.push_back(ftemp);
    }
  }

  // Convert vector of float to string (for writing MFCC file output)
  std::string v_d_to_string(v_d_t vec) {
    std::stringstream vecStream;
    for (int i = 0; i < vec.size() - 1; i++) {
      vecStream << std::scientific << vec[i];
      vecStream << ", ";
    }
    vecStream << std::scientific << vec.back();
    vecStream << "\n";
    return vecStream.str();
  }

 public:
  // MFCC class constructor
  MFCC(int sampFreq = 5333, int nCep = 20, int winLength = 50,
       int frameShift = 50, int numFilt = 12, float lf = 50, float hf = 2666) {
    fs = sampFreq;         // Sampling frequency
    numCepstra = nCep;     // Number of cepstra
    numFilters = numFilt;  // Number of Mel warped filters
    preEmphCoef = 0.97;    // Pre-emphasis coefficient
    lowFreq = lf;          // Filterbank low frequency cutoff in Hertz
    highFreq = hf;         // Filterbank high frequency cutoff in Hertz
    numFFT = fs <= 20000 ? 512 : 2048;            // FFT size
    winLengthSamples = 1 + winLength * fs / 1e3;  // winLength in milliseconds
    frameShiftSamples =
        1 + frameShift * fs / 1e3;  // frameShift in milliseconds

    numFFTBins = numFFT / 2 + 1;
    powerSpectralCoef.assign(numFFTBins, 0);
    prevsamples.assign(winLengthSamples - frameShiftSamples, 0);

    initFilterbank();
    initHamDct();
    compTwiddle();
  }

  // Process each frame and extract MFCC
  std::string processFrame(int16_t* samples, size_t N) {
    // Add samples from the previous frame that overlap with the current frame
    // to the current samples and create the frame.
    frame = prevsamples;
    for (int i = 0; i < N; i++) {
      frame.push_back(samples[i]);
    }
    prevsamples.assign(frame.begin() + frameShiftSamples, frame.end());

    preEmphHam();
    computePowerSpec();
    applyLMFB();
    applyDct();

    return v_d_to_string(mfcc);
  }
  template <typename T>
  void processFrame(v_d_t& samples, T& v_f) {
    // Add samples from the previous frame that overlap with the current frame
    // to the current samples and create the frame.
    frame = prevsamples;
    for (int i = 0; i < samples.size(); i++) {
      frame.push_back(samples[i]);
    }
    prevsamples.assign(frame.begin() + frameShiftSamples, frame.end());
    preEmphHam();
    computePowerSpec();
    applyLMFB();
    applyDct();
    for (auto it = mfcc.cbegin(); it != mfcc.cend(); it++) {
      v_f.push_back(static_cast<float>(*it));
    }
  }
  template <typename T>
  void processFrame(int16_t* samples, size_t N, T& v_f) {
    // Add samples from the previous frame that overlap with the current frame
    // to the current samples and create the frame.
    frame = prevsamples;
    for (int i = 0; i < N; i++) {
      frame.push_back(samples[i]);
    }
    prevsamples.assign(frame.begin() + frameShiftSamples, frame.end());
    preEmphHam();
    computePowerSpec();
    applyLMFB();
    applyDct();
    for (auto it = mfcc.cbegin(); it != mfcc.cend(); it++) {
      v_f.push_back(static_cast<float>(*it));
    }
  }
  // Read input file stream, extract MFCCs and write to output file stream
  int process(std::ifstream& wavFp, std::ofstream& mfcFp) {
    // Read the wav header
    wavHeader hdr;
    int headerSize = sizeof(wavHeader);
    wavFp.read((char*)&hdr, headerSize);

    // Check audio format
    if (hdr.AudioFormat != 1 || hdr.bitsPerSample != 16) {
      // std::cerr << "Unsupported audio format, use 16 bit PCM Wave" <<
      // std::endl;
      return 1;
    }
    // Check sampling rate
    if (hdr.SamplesPerSec != fs) {
      // std::cerr << "Sampling rate mismatch: Found " << hdr.SamplesPerSec << "
      // instead of " << fs <<std::endl;
      return 1;
    }

    // Check sampling rate
    if (hdr.NumOfChan != 1) {
      // std::cerr << hdr.NumOfChan << " channel files are unsupported. Use
      // mono." <<std::endl;
      return 1;
    }

    // Initialise buffer
    uint16_t bufferLength = winLengthSamples - frameShiftSamples;
    int16_t* buffer = new int16_t[bufferLength];
    int bufferBPS = (sizeof buffer[0]);

    // Read and set the initial samples
    wavFp.read((char*)buffer, bufferLength * bufferBPS);
    for (int i = 0; i < bufferLength; i++) prevsamples[i] = buffer[i];
    delete[] buffer;

    // Recalculate buffer size
    bufferLength = frameShiftSamples;
    buffer = new int16_t[bufferLength];

    // Read data and process each frame
    wavFp.read((char*)buffer, bufferLength * bufferBPS);
    while (wavFp.gcount() == bufferLength * bufferBPS && !wavFp.eof()) {
      mfcFp << processFrame(buffer, bufferLength);
      wavFp.read((char*)buffer, bufferLength * bufferBPS);
    }
    delete[] buffer;
    buffer = nullptr;
    return 0;
  }
};
};      // namespace Math
#endif  //__MFCC_H
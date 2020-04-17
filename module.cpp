#include <pybind11/pybind11.h>

#include "MFCC.h"
namespace py = pybind11;
py::list process_impl(py::list &inList, int &sampFreq, int &nCep,
                      int &winLength, int &frameShift, int &numFilt, float &lf,
                      float &hf) {
  int kMFCCWinLength = winLength;  // 48 96 182
  int kMFCCFrameShift = frameShift;
  unsigned int WINDOW_SIZE =
      static_cast<unsigned int>(kMFCCWinLength * 512 / 96.0);
  unsigned int HOP_SIZE =
      static_cast<unsigned int>(kMFCCFrameShift * 512 / 96.0);
  py::list &li = inList;
  py::list lo;
  Math::v_d_t out_vector;
  int16_t *tmpbuf = new int16_t[WINDOW_SIZE];
  Math::MFCC *mfcc_instance =
      new Math::MFCC(sampFreq, nCep, winLength, frameShift, numFilt, lf, hf);
  for (int i = 0; i <= inList.size() - WINDOW_SIZE; i += HOP_SIZE) {
    for (int j = i; j < i + WINDOW_SIZE; j++) {
      tmpbuf[j - i] = py::cast<int16_t>(li[j]);
    }
    out_vector.clear();
    mfcc_instance->processFrame<Math::v_d_t>(tmpbuf, WINDOW_SIZE, out_vector);
    py::list tmp;
    for (auto it = out_vector.cbegin(); it != out_vector.cend(); ++it) {
      tmp.append(py::float_(*it));
    }
    lo.append(tmp);
  }
  delete[] tmpbuf;
  delete mfcc_instance;
  return lo;
}

PYBIND11_MODULE(bearmfcc, m) {
  m.doc() = "pybind11 mfcc plugin";  // optional module docstring
  m.def("process", &process_impl, "MFCC C++ implementation", py::arg("inList"),
        py::arg("sampFreq") = 5333, py::arg("nCep") = 16,
        py::arg("winLength") = 96, py::arg("frameShift") = 96,
        py::arg("numFilt") = 12, py::arg("lf") = 50, py::arg("hf") = 2666);
}
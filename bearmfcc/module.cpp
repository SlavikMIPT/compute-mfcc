#include <pybind11/pybind11.h>

#include "MFCC.h"
namespace py = pybind11;
py::list process_impl(py::list &in_list, int sampFreq = 5333,
                            int nCep = 16, int winLength = 96,
                            int frameShift = 96, int numFilt = 12,
                            float lf = 50, float hf = 2666) {
    int kMFCCWinLength = winLength; //48 96 182 
    int kMFCCFrameShift = frameShift;
    unsigned int WINDOW_SIZE = static_cast<unsigned int>(kMFCCWinLength * 512 / 96.0);
    unsigned int HOP_SIZE = static_cast<unsigned int>(kMFCCWinLength * 512 / 96.0);
  py::list li = in_list, lo;
  Math::v_d_t frame, out_vector;
  Math::MFCC mfcc_instance(sampFreq, nCep, winLength, frameShift, numFilt, lf,
                           hf);
  for (int i = 0; i <= in_list.size() - WINDOW_SIZE; i += HOP_SIZE) {
      frame.clear();
      for (int j=i; j < i + WINDOW_SIZE; j++)
      {
          frame.push_back(li[j].cast<double>());
      }
      out_vector.clear();
      mfcc_instance.processFrame<Math::v_d_t>(frame, out_vector);
      py::list tmp;
      for (auto it = out_vector.cbegin(); it != out_vector.cend(); ++it)
      {
          tmp.append(py::float_(*it));
      }
      lo.append(tmp);
  }

  return lo;
}

PYBIND11_MODULE(bearmfcc, m) {
  m.doc() = "pybind11 mfcc plugin";  // optional module docstring
  m.def("process", &process_impl,
        "");
}
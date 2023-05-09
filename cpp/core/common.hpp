#include <cstdint> // int64_t

namespace distblas::core {

template<typename T>
struct Tuple {
  int64_t row;
  int64_t col;
  T value;
};

template<typename T>
struct CSR {
  int64_t row;
  int64_t col;
  T value;
};


}

#pragma once
#include <mpi.h>
#include <string>
#include <vector>
#include "Common.hpp"

using namespace std;
using namespace distblas::core;
namespace distblas::io {
/**
 * This class implements IO operations of DistBlas library.
 */
template <typename T>
class ParallelIO {
private:

public:

  ParallelIO();
  ~ParallelIO();

  /**
   * Interface for parallel reading of Matrix Market formatted files
   * @param file_path
   */
  void parallel_read_MM(string file_path, vector<Tuple<T>> &coords);
};
} // namespace distblas::io

#include "../../io/CombBLASIOReader.tpp"
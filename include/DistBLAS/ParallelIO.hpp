#pragma once
#include <mpi.h>
#include <string>
#include <vector>
#include "Tuple.hpp"

using namespace std;
using namespace distblas::core;
namespace distblas::io {
/**
 * This class implements IO operations of DistBlas library.
 */
class ParallelIO {
private:

public:
  ParallelIO();
  ~ParallelIO();

  /**
   * Interface for parallel reading of Matrix Market formatted files
   * @param file_path
   */
  vector<Tuple> parallel_read_MM(string file_path);
};
} // namespace distblas::io

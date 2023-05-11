#pragma once
#include "../core/sparse_mat.hpp"

using namespace std;
using namespace distblas::core;

namespace distblas::partition  {

  class Partitioner {

  public:
    MPI_Comm comm;
    Partitioner(MPI_Comm comm);
    ~Partitioner();

    template<typename T>
    void redistribute_data(int test);
  };
}

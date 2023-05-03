#pragma once
#include "SpMat.hpp"

using namespace std;
namespace distblas::core  {

  class Partitioner {

  public:
    MPI_Comm comm;
    Partitioner(MPI_Comm comm);
    ~Partitioner();

    template <typename T>
    SpMat<T>* redistribute_data(SpMat<T> &distributed_mat, bool  inplace);


  };
}

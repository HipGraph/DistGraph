#pragma once
#include "../data_structures/sparse_mat.hpp"

using namespace std;
namespace distblas::partition  {

  class Partitioner {

  public:
    MPI_Comm comm;
    Partitioner(MPI_Comm comm);
    ~Partitioner();

    template <typename T>
    SpMat<T>* redistribute_data(SpMat<T> &distributed_mat, bool  inplace);


  };
}

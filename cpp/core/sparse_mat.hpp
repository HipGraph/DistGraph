#pragma once
#include <iostream>
#include <mpi.h>
#include <vector>
#include "../../utils/Common.hpp"
#include "distributed_mat.hpp"

using namespace std;

namespace distblas::core {

/**
 * This class represents the Sparse Matrix
 */

template <typename T>
class SpMat: public DistributedMat {

public:
  int gRows, gCols, gNNz;
  vector<Tuple<T>> coords;

  /**
   * Constructor for Sparse Matrix representation of  Adj matrix
   * @param coords  (src, dst, value) Tuple vector loaded as input
   * @param gRows   total number of Rows in Distributed global Adj matrix
   * @param gCols   total number of Cols in Distributed global Adj matrix
   * @param gNNz     total number of NNz in Distributed global Adj matrix
   */
  SpMat(vector<Tuple<T>> &coords, int gRows, int gCols, int gNNz);

  void initialize_csr();


  ~SpMat();


};

}

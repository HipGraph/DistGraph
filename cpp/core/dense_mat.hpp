#pragma once
#include "common.h"
#include "distributed_mat.hpp"
#include <Eigen/Dense>
#include <memory>
#include <random>
#include <unordered_map>

using namespace std;
using namespace Eigen;

namespace distblas::core {

/**
 * This class wraps the Eigen/Dense matrix and represents
 * local dense matrix.
 */
template <typename DENT> class DenseMat : DistributedMat {

private:
  unique_ptr<Matrix<DENT, Dynamic, Dynamic>> matrixPtr;

public:
  uint64_t rows;
  uint64_t cols;
  /**
   * create matrix with random initialization
   * @param rows Number of rows of the matrix
   * @param cols Number of cols of the matrix
   */
  DenseMat(uint64_t rows, uint64_t cols) {

    this->matrixPtr = make_unique<Matrix<DENT, Dynamic, Dynamic>>(rows, cols);
    this->rows = rows;
    this->cols = cols;
  }

  /**
   *
   * @param rows Number of rows of the matrix
   * @param cols  Number of cols of the matrix
   * @param init_mean  initialize with normal distribution with given mean
   * @param std  initialize with normal distribution with given standard
   * deviation
   */
  DenseMat(uint64_t rows, uint64_t cols, double init_mean, double std) {
    this->rows = rows;
    this->cols = cols;
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> distribution(init_mean, std);
    this->matrixPtr = make_unique<Matrix<DENT, Dynamic, Dynamic>>(rows, cols);
    (*this->matrixPtr).setRandom();
    if (init_mean != 0.0 or std != 1.0) {
#pragma omp parallel
      for (int i = 0; i < (*this->matrixPtr)->rows(); ++i) {
        for (int j = 0; j < (*this->matrixPtr)->cols(); ++j) {
          (*this->matrixPtr)(i, j) = distribution(
              gen); // Generate random value with custom distribution
        }
      }
    }
  }

  ~DenseMat() {}

  void print_matrix() {
    for (int i = 0; i < (*this->matrixPtr).rows(); ++i) {
      for (int j = 0; j < (*this->matrixPtr).cols(); ++j) {
        cout << (*this->matrixPtr)(i, j) << " ";
      }
      cout << endl;
    }
  }
};

} // namespace distblas::core

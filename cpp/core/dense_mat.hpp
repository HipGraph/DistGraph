#pragma once
#include "common.h"
#include "distributed_mat.hpp"
#include <Eigen/Dense>
#include <memory>
#include <random>

using namespace std;
using Eigen::MatrixXd;

namespace distblas::core {

/**
 * This class wraps the Eigen/Dense matrix and represents
 * local dense matrix.
 */
class DenseMat : DistributedMat {

private:
  unique_ptr<MatrixXd> matrixPtr;
  uint64_t rows;
  uint64_t cols;

public:
  /**
   * create matrix with random initialization
   * @param rows Number of rows of the matrix
   * @param cols Number of cols of the matrix
   */
  DenseMat(uint64_t rows, uint64_t cols) {

    this->matrixPtr = std::make_unique<MatrixXd>(MatrixXd::Random(rows, cols));
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
    Eigen::MatrixXd matrixL(rows, cols);
    matrixL.setRandom();
    if (init_mean != 0.0 or std != 1.0 ) {
#pragma omp parallel
      for (int i = 0; i < matrixL.rows(); ++i) {
        for (int j = 0; j < matrixL.cols(); ++j) {
          matrixL(i, j) = distribution(gen); // Generate random value with custom distribution
        }
      }
    }
    this->matrixPtr = unique_ptr<MatrixXd>(matrixL);
  }

  ~DenseMat() {}

  void print_matrix() {

    for (int i = 0; i < (this->matrixPtr.get()).rows(); ++i) {
      for (int j = 0; j < (this->matrixPtr.get()).cols(); ++j) {
        cout << (this->matrixPtr.get())(i, j) << " ";
      }
      cout << endl;
    }

  }
};

} // namespace distblas::core

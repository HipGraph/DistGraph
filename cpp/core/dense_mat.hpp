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
template <typename DENT, size_t embedding_dim> class DenseMat : DistributedMat {

private:
  unique_ptr<Matrix<DENT, Dynamic, embedding_dim>> matrixPtr;
  unique_ptr<vector<unordered_map<uint64_t,Matrix<DENT, embedding_dim, 1>>>> cachePtr;

public:
  uint64_t rows;
  /**
   * create matrix with random initialization
   * @param rows Number of rows of the matrix
   * @param cols Number of cols of the matrix
   */
  DenseMat(uint64_t rows, int world_size) {

    this->matrixPtr = make_unique<Matrix<DENT, Dynamic, embedding_dim>>(rows, embedding_dim);
    this->cachePtr =
        std::make_unique<std::vector<std::unordered_map<uint64_t, Eigen::Matrix<DENT, embedding_dim, 1>>>>(world_size);

    this->rows = rows;
  }

  /**
   *
   * @param rows Number of rows of the matrix
   * @param cols  Number of cols of the matrix
   * @param init_mean  initialize with normal distribution with given mean
   * @param std  initialize with normal distribution with given standard
   * deviation
   */
  DenseMat(uint64_t rows, double init_mean, double std, int world_size) {
    this->rows = rows;
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> distribution(init_mean, std);
    this->matrixPtr = make_unique<Matrix<DENT, Dynamic, embedding_dim>>(rows,embedding_dim);
    this->cachePtr =
        std::make_unique<std::vector<std::unordered_map<uint64_t,
                                                        Eigen::Matrix<DENT,
                                                                      embedding_dim,
                                                                      1>>>>(world_size);
    (*this->matrixPtr).setRandom();
    if (init_mean != 0.0 or std != 1.0) {
#pragma omp parallel
      for (int i = 0; i < (*this->matrixPtr).rows(); ++i) {
        for (int j = 0; j < (*this->matrixPtr).cols(); ++j) {
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

  void insert_cache(int rank, int key, std::array<DENT, embedding_dim> &arr) {
    Map<Matrix<DENT, Eigen::Dynamic, 1>> eigenVector(arr.data(), embedding_dim);
    (*this->cachePtr)[rank].insert_or_assign(key,eigenVector);
  }

  Matrix<DENT, embedding_dim, 1> fetch_data_vector_from_cache(int rank, int key) {
    return  (*this->cachePtr)[rank][key];
  }

  std::array<DENT, embedding_dim> fetch_local_data(int local_key) {
  std::array<DENT, embedding_dim> stdArray;
  Eigen::Map<Eigen::Matrix<DENT, 1, embedding_dim>>(stdArray.data()) = (*this->matrixPtr).row(local_key);
  return stdArray;

  }
};

} // namespace distblas::core

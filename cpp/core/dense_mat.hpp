#pragma once
#include "common.h"
#include "distributed_mat.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_map>
#include <fstream>
#include <mpi.h>

using namespace std;
using namespace Eigen;

namespace distblas::core {

/**
 * This class wraps the Eigen/Dense matrix and represents
 * local dense matrix.
 */
template <typename DENT, size_t embedding_dim> class DenseMat : DistributedMat {

private:

  unique_ptr<vector<unordered_map<uint64_t, Matrix<DENT, embedding_dim, 1>>>>
      cachePtr;

public:
  uint64_t rows;
  unique_ptr<Matrix<DENT, Dynamic, embedding_dim>> matrixPtr;
  /**
   * create matrix with random initialization
   * @param rows Number of rows of the matrix
   * @param cols Number of cols of the matrix
   */
  DenseMat(uint64_t rows, int world_size) {

    this->matrixPtr =
        make_unique<Matrix<DENT, Dynamic, embedding_dim>>(rows, embedding_dim);
    this->cachePtr = std::make_unique<std::vector<
        std::unordered_map<uint64_t, Eigen::Matrix<DENT, embedding_dim, 1>>>>(
        world_size);

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
    this->matrixPtr =
        make_unique<Matrix<DENT, Dynamic, embedding_dim>>(rows, embedding_dim);
    this->cachePtr = std::make_unique<std::vector<
        std::unordered_map<uint64_t, Eigen::Matrix<DENT, embedding_dim, 1>>>>(
        world_size);
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
    (*this->cachePtr)[rank].insert_or_assign(key, eigenVector);
  }

  Matrix<DENT, embedding_dim, 1> fetch_data_vector_from_cache(int rank,
                                                              int key) {
    return (*this->cachePtr)[rank][key];
  }

  std::array<DENT, embedding_dim> fetch_local_data(int local_key) {
    std::array<DENT, embedding_dim> stdArray;
    Eigen::Matrix<DENT, Eigen::Dynamic, embedding_dim>& matrix = *this->matrixPtr;
    Eigen::Array<DENT, 1, embedding_dim> eigenArray = matrix.row(local_key).transpose().array();
    std::copy(eigenArray.data(), eigenArray.data() + embedding_dim, stdArray.data());
    return stdArray;
  }

  Eigen::Array<DENT, 1, embedding_dim> fetch_local_eigen_vector(int local_key) {
    Eigen::Matrix<DENT, Eigen::Dynamic, embedding_dim>& matrix = *this->matrixPtr;
    return matrix.row(local_key);
  }


  void print_cache() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for(int i=0;i<(*this->cachePtr).size();i++){
      unordered_map<uint64_t, Matrix<DENT, embedding_dim, 1>> map = (*this->cachePtr)[i];

      string output_path = "rank_"+to_string(rank)+"remote_rank_" + to_string(i) +  ".txt";
      char stats[500];
      strcpy(stats, output_path.c_str());
      ofstream fout(stats, std::ios_base::app);


      for (const auto& kvp : map) {
        uint64_t key = kvp.first;
        const Eigen::Matrix<DENT, embedding_dim, 1>& value = kvp.second;
        fout<<key<<" ";
        for (int i = 0; i < embedding_dim; ++i) {
          fout << value(i) << " ";
        }
        fout<< std::endl;
      }
    }
  }


};

} // namespace distblas::core

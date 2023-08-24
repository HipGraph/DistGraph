#pragma once
#include "../net/process_3D_grid.hpp"
#include "common.h"
#include "distributed_mat.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <random>
#include <unordered_map>
#include "sparse_mat.hpp"

using namespace std;
using namespace Eigen;
using namespace distblas::net;

namespace distblas::core {

/**
 * This class wraps the Eigen/Dense matrix and represents
 * local dense matrix.
 */
template <typename SPT, typename DENT, size_t embedding_dim> class DenseMat : DistributedMat {

private:
public:
  uint64_t rows;
  unique_ptr<Matrix<DENT, Dynamic, embedding_dim>> matrixPtr;
  unique_ptr<vector<unordered_map<uint64_t, std::array<DENT, embedding_dim>>>>
      cachePtr;
  DENT *nCoordinates;
  SpMat<SPT> *sp_local;
  Process3DGrid *grid;
  /**
   * create matrix with random initialization
   * @param rows Number of rows of the matrix
   * @param cols Number of cols of the matrix
   */
  DenseMat(uint64_t rows, int world_size) {
    this->matrixPtr =
        make_unique<Matrix<DENT, Dynamic, embedding_dim>>(rows, embedding_dim);
    this->cachePtr = std::make_unique<std::vector<
        std::unordered_map<uint64_t, std::array<DENT, embedding_dim>>>>(
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
  DenseMat(SpMat<SPT> *sp_local, Process3DGrid *grid,
           uint64_t rows, double init_mean, double std) {

    this->rows = rows;
    this->sp_local = sp_local;
    this->grid = grid;
    this->cachePtr = std::make_unique<std::vector<
        std::unordered_map<uint64_t, std::array<DENT, embedding_dim>>>>(
        grid->world_size);
    nCoordinates =
        static_cast<DENT *>(::operator new(sizeof(DENT[rows * embedding_dim])));
    std::srand(this->grid->global_rank);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < embedding_dim; j++) {
        DENT val = -1.0 + 2.0 * rand() / (RAND_MAX + 1.0);
        nCoordinates[i * embedding_dim + j] = val;
      }
    }
//    this->initialize_cache();
  }

  ~DenseMat() {}

  void print_matrix() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    string output_path = "embedding" + to_string(rank) + ".txt";
    char stats[500];
    strcpy(stats, output_path.c_str());
    ofstream fout(stats, std::ios_base::app);
    for (int i = 0; i < rows; ++i) {
      fout << (i + 1) << " ";
      for (int j = 0; j < embedding_dim; ++j) {
        fout << nCoordinates[i * embedding_dim + j] << " ";
      }
      fout << endl;
    }
  }

  void print_matrix_rowptr(int iter) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    string output_path = "itr_" + to_string(iter) + "_embedding.txt";
    char stats[500];
    strcpy(stats, output_path.c_str());
    ofstream fout(stats, std::ios_base::app);
    //    fout << (*this->matrixPtr).rows() << " " << (*this->matrixPtr).cols()
    //         << endl;
    for (int i = 0; i < rows; ++i) {
      fout << i + 1 + rank * rows << " ";
      for (int j = 0; j < embedding_dim; ++j) {
        fout << this->nCoordinates[i * embedding_dim + j] << " ";
      }
      fout << endl;
    }
  }

  void insert_cache(int rank, uint64_t key,
                    std::array<DENT, embedding_dim> &arr) {
    //    Map<Matrix<DENT, Eigen::Dynamic, 1>> eigenVector(arr.data(),
    //    embedding_dim);
    (*this->cachePtr)[rank].insert_or_assign(key, arr);
  }

  //  Matrix<DENT, embedding_dim, 1> fetch_data_vector_from_cache(int rank,
  //                                                              int key) {
  //    return (*this->cachePtr)[rank][key];
  //  }

  std::array<DENT, embedding_dim> fetch_data_vector_from_cache(int rank,
                                                               uint64_t key) {
    //    int my_rank;
    //    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    //    if(this->searchForKey(key)) {
    return (*this->cachePtr)[rank][key];
    //    }else {
    //      cout<<" rank "<< my_rank<<"Error cannot find key"<<key<<" for rank
    //      "<< rank<<endl;
    //    }
  }

  std::array<DENT, embedding_dim> fetch_local_data(int local_key) {
    std::array<DENT, embedding_dim> stdArray;

    int base_index = local_key * embedding_dim;

    //    Eigen::Matrix<DENT, Eigen::Dynamic, embedding_dim>& matrix =
    //    *this->matrixPtr; Eigen::Array<DENT, 1, embedding_dim> eigenArray =
    //    matrix.row(local_key).transpose().array();
    std::copy(nCoordinates + base_index,
              nCoordinates + base_index + embedding_dim, stdArray.data());
    return stdArray;
  }

  Eigen::Array<DENT, 1, embedding_dim> fetch_local_eigen_vector(int local_key) {
    Eigen::Matrix<DENT, Eigen::Dynamic, embedding_dim> &matrix =
        *this->matrixPtr;
    return matrix.row(local_key);
  }

  void initialize_cache() {
    CSRLinkedList<SPT> *batch_list = (this->sp_local)->get_batch_list(0);
    auto head = batch_list->getHeadNode();
    CSRLocal<SPT> *csr_block_local = (head.get())->data.get();
    CSRLocal<SPT> *csr_block_remote = nullptr;
    if (this->grid->world_size > 1) {
      auto remote = (head.get())->next;
      csr_block_remote = (remote.get())->data.get();
      CSRHandle *csr_handle = csr_block_remote->handler.get();
      vector<double> values = csr_handle->values;
      cout<<" "<<grid->global_rank<<" remote CSR block accessed success"<<endl;
      # pragma omp parallel for
      for (int i = 0; i < this->grid->world_size; i++) {
        if (i != this->grid->global_rank) {
          std::srand(i);
          for (uint64_t j = 0; j < this->sp_local->proc_row_width; j++) {

            auto global_index = j + i*this->sp_local->proc_row_width;

            auto result = std::find(values.begin(), values.end(), global_index);
            std::array<DENT, embedding_dim> stdArray;

            for (int k = 0; k < embedding_dim; k++) {
              DENT val = -1.0 + 2.0 * rand() / (RAND_MAX + 1.0);
              stdArray[k]=val;
            }

            if (result != values.end()) {
              this->insert_cache(i,global_index,stdArray);
            }
          }
        }
      }
    }
  }

  void print_cache(int iter) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < (*this->cachePtr).size(); i++) {
      unordered_map<uint64_t, std::array<DENT, embedding_dim>> map =
          (*this->cachePtr)[i];

      string output_path = "rank_" + to_string(rank) + "remote_rank_" +
                           to_string(i) + " itr_" + to_string(iter) + ".txt";
      char stats[500];
      strcpy(stats, output_path.c_str());
      ofstream fout(stats, std::ios_base::app);

      for (const auto &kvp : map) {
        uint64_t key = kvp.first;
        const std::array<DENT, embedding_dim> &value = kvp.second;
        fout << key << " ";
        for (int i = 0; i < embedding_dim; ++i) {
          fout << value[i] << " ";
        }
        fout << std::endl;
      }
    }
  }

  bool searchForKey(uint64_t key) {
    for (const auto &nestedMap : *cachePtr) {
      auto it = nestedMap.find(key);
      if (it != nestedMap.end()) {
        auto result = it->second;
        return true; // Key found in the current nestedMap
      }
    }
    return false; // Key not found in any nestedMap
  }
};

} // namespace distblas::core

#pragma once
#include "../net/process_3D_grid.hpp"
#include "common.h"
#include "distributed_mat.hpp"
#include "sparse_mat.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <random>
#include <unordered_map>

using namespace std;
using namespace Eigen;
using namespace distblas::net;

namespace distblas::core {

/**
 * This class wraps the Eigen/Dense matrix and represents
 * local dense matrix.
 */
template <typename SPT, typename DENT, size_t embedding_dim>
class DenseMat : DistributedMat {

private:
public:
  uint64_t rows;
  unique_ptr<vector<unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>>>>
      cachePtr;
  DENT *nCoordinates;
  Process3DGrid *grid;

  /**
   *
   * @param rows Number of rows of the matrix
   * @param cols  Number of cols of the matrix
   * @param init_mean  initialize with normal distribution with given mean
   * @param std  initialize with normal distribution with given standard
   * deviation
   */
  DenseMat(Process3DGrid *grid, uint64_t rows) {

    this->rows = rows;
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
  }

  ~DenseMat() {}

  void insert_cache(int rank, uint64_t key, int batch_id, int iteration,
                    std::array<DENT, embedding_dim> &arr) {
    distblas::core::CacheEntry<DENT, embedding_dim> entry;
    entry.batch_id = batch_id;
    entry.iteration = iteration;
    entry.value = arr;
    (*this->cachePtr)[rank].insert_or_assign(key, entry);
  }

  DENT *fetch_data_vector_from_cache(int rank, uint64_t key) {

    // Access the array using the provided rank and key
    auto &arrayMap = (*cachePtr)[rank];
    auto it = arrayMap.find(key);

    if (it != arrayMap.end()) {
      return it->second.value.data(); // Pointer to the array's data
    } else {
      return nullptr; // Key not found
    }
  }

  std::array<DENT, embedding_dim> fetch_local_data(int local_key) {
    std::array<DENT, embedding_dim> stdArray;

    int base_index = local_key * embedding_dim;
    std::copy(nCoordinates + base_index,
              nCoordinates + base_index + embedding_dim, stdArray.data());
    return stdArray;
  }

  void invalidate_cache(int current_itr, int current_batch) {
    for (int i = 0; i < grid->world_size; i++) {
      auto &arrayMap = (*cachePtr)[i];
      for (auto it = arrayMap.begin(); it != arrayMap.end();) {
        distblas::core::CacheEntry<DENT, embedding_dim> cache_ent = it->second;
        if (cache_ent.inserted_itr < current_itr and
            cache_ent.inserted_batch_id <= current_batch) {
          it = arrayMap.erase(it);
        } else {
          // Move to the next item
          ++it;
        }
      }
    }
  }

  // Utitly methods
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

#pragma once
#include "../net/process_3D_grid.hpp"
#include "common.h"
#include "distributed_mat.hpp"
#include "sparse_mat.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <random>
#include <unordered_map>

using namespace std;

using namespace distblas::net;

namespace distblas::core {

/**
 * This class represents  the dense matrix.
 */
template <typename INDEX_TYPE, typename VALUE_TYPE, size_t embedding_dim>
class DenseMat : public DistributedMat {

private:
public:
  uint64_t rows;
  uint64_t cols=0;
  unique_ptr<vector<unordered_map<INDEX_TYPE, CacheEntry<VALUE_TYPE, embedding_dim>>>>
      cachePtr;
  unique_ptr<vector<unordered_map<INDEX_TYPE, CacheEntry<VALUE_TYPE, embedding_dim>>>>
      tempCachePtr;
  Process3DGrid *grid;
  VALUE_TYPE *nCoordinates;
  /**
   *
   * @param rows Number of rows of the matrix
   * @param cols  Number of cols of the matrix
   * @param init_mean  initialize with normal distribution with given mean
   * @param std  initialize with normal distribution with given standard
   * deviation
   */
  DenseMat(Process3DGrid *grid, INDEX_TYPE rows): DistributedMat() {

    this->rows = rows;
    this->grid = grid;
    this->cols=embedding_dim;

    cout<<" col world size "<<grid->col_world_size<<endl;

    this->cachePtr = std::make_unique<std::vector<
        std::unordered_map<INDEX_TYPE, CacheEntry<VALUE_TYPE, embedding_dim>>>>(
        grid->col_world_size);
    this->tempCachePtr = std::make_unique<std::vector<
        std::unordered_map<INDEX_TYPE, CacheEntry<VALUE_TYPE, embedding_dim>>>>(
        grid->col_world_size);
    this->nCoordinates =
        static_cast<VALUE_TYPE *>(::operator new(sizeof(VALUE_TYPE[rows * embedding_dim])));
//    std::srand(this->grid->global_rank);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < embedding_dim; j++) {
        VALUE_TYPE val = -1.0 + 2.0 * rand() / (RAND_MAX + 1.0);
        this->nCoordinates[i * embedding_dim + j] = val;
      }
    }
    this->nnz_count = make_unique<vector<INDEX_TYPE>>(rows,0);
    this->state_metadata = make_unique<vector<vector<VALUE_TYPE>>>(rows,vector<VALUE_TYPE>(embedding_dim,0));

  }

  ~DenseMat() {}

    DenseMat(Process3DGrid *grid, INDEX_TYPE rows, INDEX_TYPE cols, bool lazy=false): DistributedMat() {

        this->rows = rows;
        this->grid = grid;
        this->cols = cols;
        cout<<" rows "<<rows<<" cols "<<cols<<" total"<<rows*cols<<endl;
        this->nCoordinates = make_unique<vector<VALUE_TYPE>>(rows*cols)->data();
//    std::srand(this->grid->global_rank);
        if (!lazy) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    VALUE_TYPE val = -1.0 + 2.0 * rand() / (RAND_MAX + 1.0);
                    this->nCoordinates[i * cols + j] = val;
                }
            }
        }

    }




  void insert_cache(int rank, INDEX_TYPE key, int batch_id, int iteration,
                    std::array<VALUE_TYPE, embedding_dim> &arr, bool temp) {
    distblas::core::CacheEntry<VALUE_TYPE, embedding_dim> entry;
    entry.inserted_batch_id = batch_id;
    entry.inserted_itr = iteration;
    entry.value = arr;
    if (temp) {
      (*this->tempCachePtr)[rank][key]= entry;
    } else {
      (*this->cachePtr)[rank][key]= entry;
    }
  }

  auto fetch_data_vector_from_cache(std::array<VALUE_TYPE, embedding_dim> &value,int rank, INDEX_TYPE key, bool temp) {

    // Access the array using the provided rank and key

    auto arrayMap = (temp) ? (*tempCachePtr)[rank] : (*cachePtr)[rank];
    auto it = arrayMap.find(key);

    if (it != arrayMap.end()) {
      auto temp = it->second;
      value =  temp.value;
    }else {
      throw std::runtime_error("cannot find the given key");
    }

  }

  std::array<VALUE_TYPE, embedding_dim> fetch_local_data(int local_key) {
    std::array<VALUE_TYPE, embedding_dim> stdArray;

    int base_index = local_key * embedding_dim;
    std::copy(nCoordinates + base_index,
              this->nCoordinates + base_index + embedding_dim, stdArray.data());
    return stdArray;
  }

  void invalidate_cache(int current_itr, int current_batch, bool temp) {
    if (temp) {
      purge_temp_cache();
    } else {
      for (int i = 0; i < grid->col_world_size; i++) {
        auto &arrayMap = (*cachePtr)[i];
        for (auto it = arrayMap.begin(); it != arrayMap.end();) {
          distblas::core::CacheEntry<VALUE_TYPE, embedding_dim> cache_ent =
              it->second;
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
  }

  void purge_temp_cache() {
    for (int i = 0; i < grid->col_world_size; i++) {
      (*this->tempCachePtr)[i].clear();
      std::unordered_map<INDEX_TYPE, CacheEntry<VALUE_TYPE, embedding_dim>>().swap(
          (*this->tempCachePtr)[i]);
    }
  }

  // Utitly methods
  void print_matrix() {
    int rank = grid->rank_in_col;
    string output_path = "embedding" + to_string(rank) + ".txt";
    char stats[500];
    strcpy(stats, output_path.c_str());
    ofstream fout(stats, std::ios_base::app);
    for (int i = 0; i < rows; ++i) {
      fout << (i + 1) << " ";
      for (int j = 0; j < embedding_dim; ++j) {
        fout << this->nCoordinates[i * embedding_dim + j] << " ";
      }
      fout << endl;
    }
  }

  void print_matrix_rowptr(int iter) {
    int rank= grid->rank_in_col;
    string output_path =
        "rank_" + to_string(rank) + "itr_" + to_string(iter) + "_embedding.txt";
    char stats[500];
    strcpy(stats, output_path.c_str());
    ofstream fout(stats, std::ios_base::app);
    //    fout << (*this->matrixPtr).rows() << " " << (*this->matrixPtr).cols()
    //         << endl;
    for (int i = 0; i < rows; ++i) {
      fout << i  + rank * rows << " ";
      for (int j = 0; j < embedding_dim; ++j) {
        fout << this->nCoordinates[i * embedding_dim + j] << " ";
      }
      fout << endl;
    }
  }

  void print_cache(int iter) {
    int rank = grid->rank_in_col;

    for (int i = 0; i < (*this->cachePtr).size(); i++) {
      unordered_map<INDEX_TYPE, CacheEntry<VALUE_TYPE, embedding_dim>> map =
          (*this->cachePtr)[i];
//      (*this->tempCachePtr)[i];

      string output_path = "rank_" + to_string(rank) + "remote_rank_" +
                           to_string(i) + " itr_" + to_string(iter) + ".txt";
      char stats[500];
      strcpy(stats, output_path.c_str());
      ofstream fout(stats, std::ios_base::app);

      for (const auto &kvp : map) {
        INDEX_TYPE key = kvp.first;
        const std::array<VALUE_TYPE, embedding_dim> &value = kvp.second.value;
        fout << key << " ";
        for (int i = 0; i < embedding_dim; ++i) {
          fout << value[i] << " ";
        }
        fout << std::endl;
      }
    }
  }

  bool searchForKey(INDEX_TYPE key) {
    for (const auto &nestedMap : *cachePtr) {
      auto it = nestedMap.find(key);
      if (it != nestedMap.end()) {
        auto result = it->second;
        return true; // Key found in the current nestedMap
      }
    }
    return false; // Key not found in any nestedMap
  }


  void multiply(DenseMat<INDEX_TYPE,VALUE_TYPE,embedding_dim>* other, DenseMat<INDEX_TYPE,VALUE_TYPE,embedding_dim>* output){
    int cols = this->cols>0?this->cols:embedding_dim;

    assert(cols == other->rows);
    int output_size = this->rows*other->cols;
    output->nCoordinates = make_unique<vector<VALUE_TYPE>>(output_size)->data();
    output->rows=this->rows;
    output->cols=other->cols;
#pragma omp parallel for collapse(2)
    for(int i=0;i<this->rows;++i){
        for(int j=0;j<other->cols;++j){
            VALUE_TYPE value=0;
            for(int k=0;k<cols;++k){
                value+= this->nCoordinates[i*cols+k]*other->nCoordinates[k*other->cols+j];
            }
            output->nCoordinates[i*other->cols+j]=value;
        }
    }
  }

};

} // namespace distblas::core

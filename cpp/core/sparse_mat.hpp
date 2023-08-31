#pragma once
#include "common.h"
#include "csr_linked_list.hpp"
#include "csr_local.hpp"
#include "distributed_mat.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <parallel/algorithm>
#include <unordered_set>
#include <vector>

using namespace std;

namespace distblas::core {

/**
 * This class represents the Sparse Matrix
 */
template <typename T> class SpMat : public DistributedMat {

private:


public:
  uint64_t gRows, gCols, gNNz;
  vector<Tuple<T>> coords;
  int batch_size;
  int proc_col_width, proc_row_width;
  bool transpose = false;
  bool col_partitioned = false;
  unique_ptr<CSRLocal<T>> csr_local_data;
  unique_ptr<CSRLocal<T>> csr_local_data_native;

  /**
   * Constructor for Sparse Matrix representation of  Adj matrix
   * @param coords  (src, dst, value) Tuple vector loaded as input
   * @param gRows   total number of Rows in Distributed global Adj matrix
   * @param gCols   total number of Cols in Distributed global Adj matrix
   * @param gNNz     total number of NNz in Distributed global Adj matrix
   */
  SpMat(vector<Tuple<T>> &coords, uint64_t &gRows, uint64_t &gCols,
        uint64_t &gNNz, int &batch_size, int &proc_row_width,
        int &proc_col_width, bool transpose, bool col_partitioned) {
    this->gRows = gRows;
    this->gCols = gCols;
    this->gNNz = gNNz;
    this->coords = coords;
    this->batch_size = batch_size;
    this->proc_col_width = proc_col_width;
    this->proc_row_width = proc_row_width;
    this->transpose = transpose;
    this->col_partitioned = col_partitioned;
  }

  SpMat() {}

  void initialize_CSR_blocks() {

    # pragma omp parallel for
    for (uint64_t i = 0; i < coords.size(); i++) {
      if (col_partitioned) {
        coords[i].col %= proc_col_width;
      } else {
        coords[i].row %= proc_row_width;
      }
    }
    Tuple<T> *coords_ptr = coords.data();

    if (col_partitioned) {
      // This is always non-transpose col partitioned
      csr_local_data =
          make_unique<CSRLocal<T>>(gRows, proc_col_width, coords.size(),
                                   coords_ptr, coords.size(), transpose);
    } else {
      // This may have transpose and non transpose version for row partitioned
      csr_local_data =
          make_unique<CSRLocal<T>>(proc_row_width, gCols, coords.size(),
                                   coords_ptr, coords.size(), transpose);
      csr_local_data_native = make_unique<CSRLocal<T>>(proc_row_width, gCols, coords.size(),
                                                       coords_ptr, coords.size(), false);
    }
  }

  // if batch_id<0 it will fetch all the batches
  void fill_col_ids(int batch_id,
                    vector<vector<uint64_t>> &proc_to_id_mapping) {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    distblas::core::CSRHandle *handle = (csr_local_data.get())->handler.get();

    if (col_partitioned) {
#pragma omp parallel for
      for (int r = 0; r < world_size; r++) {
        uint64_t starting_index = proc_row_width * r;
        auto end_index =
            std::min(static_cast<uint64_t>((r + 1) * proc_row_width), gRows) -
            1;

        for (auto i = starting_index; i <= (end_index); i++) {

          auto eligible_col_id_start =
              (batch_id >= 0) ? batch_id * batch_size : 0;
          auto eligible_col_id_end =
              (batch_id >= 0)
                  ? std::min(static_cast<uint64_t>((batch_id + 1) * batch_size),
                             static_cast<uint64_t>(proc_col_width))
                  : proc_col_width;
          if (rank != r and (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1]; j++) {
              auto col_val = handle->col_idx[j];
              if (col_val >= eligible_col_id_start and
                  col_val < eligible_col_id_end) {
                // calculation of sender col_ids
                { proc_to_id_mapping[r].push_back(col_val); }
              }
            }
          }
        }
      }
    } else if (transpose) {

      // calculation of sender col_ids
#pragma omp parallel for
      for (int r = 0; r < world_size; r++) {
        uint64_t starting_index =
            (batch_id >= 0) ? batch_id * batch_size + proc_col_width * r
                            : proc_col_width * r;
        auto end_index =
            (batch_id >= 0)
                ? std::min(starting_index + batch_size,
                      std::min(static_cast<uint64_t>((r + 1) * proc_col_width),gCols)) -1
                : std::min(static_cast<uint64_t>((r + 1) * proc_col_width),gCols) -1;

        for (auto i = starting_index; i <= (end_index); i++) {
          if (rank != r and (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            { proc_to_id_mapping[r].push_back(i); }
          }
        }
      }
    }
  }



  void print_coords(bool trans) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    string output_path =
        "coords" + to_string(rank) + "trans" + to_string(trans) + ".txt";
    char stats[500];
    strcpy(stats, output_path.c_str());
    ofstream fout(stats, std::ios_base::app);

    for (int i = 0; i < coords.size(); i++) {
      fout << coords[i].row << " " << coords[i].value << " " << endl;
    }
  }

  ~SpMat() {}
};

} // namespace distblas::core

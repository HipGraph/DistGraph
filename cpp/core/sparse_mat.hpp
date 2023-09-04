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

#pragma omp parallel for
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
    }
  }

  // if batch_id<0 it will fetch all the batches
  void fill_col_ids(int batch_id, vector<vector<uint64_t>> &proc_to_id_mapping,
                    double alpha) {

    if (alpha == 0) {
      fill_col_ids_for_pulling(batch_id, proc_to_id_mapping);

    } else {

      fill_col_ids_for_pushing(batch_id, proc_to_id_mapping, alpha);
    }
  }

  void fill_col_ids_for_pulling(int batch_id,
                                vector<vector<uint64_t>> &proc_to_id_mapping) {

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD,
                  &rank); // TODO convert this is flexible grid indexes

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    distblas::core::CSRHandle *handle = (csr_local_data.get())->handler.get();

    if (col_partitioned) {

      for (int r = 0; r < world_size; r++) {
        uint64_t starting_index = batch_id * batch_size + proc_row_width * r;
        auto end_index =
            std::min(static_cast<uint64_t>((r + 1) * proc_row_width), gRows);

        for (int i = starting_index; i < end_index; i++) {
          if (rank != r and (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1]; j++) {
              auto col_val = handle->col_idx[j];
              { proc_to_id_mapping[r].push_back(col_val); }
            }
          }
        }
      }
    } else if (transpose) {
      for (int r = 0; r < world_size; r++) {
        uint64_t starting_index = proc_col_width * r;
        auto end_index =
            std::min(static_cast<uint64_t>((r + 1) * proc_col_width), gCols);
        for (int i = starting_index; i < end_index; i++) {
          if (rank != r and
              (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1];
                 j++) {
              auto col_val = handle->col_idx[j];
              uint64_t dst_start = batch_id * batch_size;
              uint64_t dst_end_index =
                  std::min((batch_id + 1) * batch_size, proc_row_width);
              if (col_val >= dst_start and col_val < dst_end_index) {
                { proc_to_id_mapping[r].push_back(i); }
              }
            }
          }
        }
      }
    }
  }

  void fill_col_ids_for_pushing(int batch_id,
                                vector<vector<uint64_t>> &proc_to_id_mapping,
                                double alpha) {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    distblas::core::CSRHandle *handle = (csr_local_data.get())->handler.get();

    auto batches = (proc_row_width / batch_size);

    if (!(proc_row_width % batch_size == 0)) {
      batches = (proc_row_width / batch_size) + 1;
    }

    if (col_partitioned) {
      // calculation of sender col_ids
//#pragma omp parallel for
      for (int r = 0; r < world_size; r++) {
        uint64_t starting_index = proc_row_width * r;
        auto end_index =
            std::min(static_cast<uint64_t>((r + 1) * proc_row_width), gRows) -
            1;

        auto per_batch_nnz = 0;
        auto total_nnz = 0;
        auto effective_nnz = 0;
        int count = 0;
        if ( alpha >0  and  alpha < 1.0) {
           total_nnz = handle->rowStart[end_index + 1] -
                           handle->rowStart[starting_index];
           effective_nnz = alpha * total_nnz;
          per_batch_nnz = effective_nnz / batches;
          starting_index =
              (batch_id < batches - 1)
                  ? (batch_id + 1) * batch_size + proc_row_width * r
                  : proc_row_width * r;
        }

        for (auto i = starting_index; i <= (end_index); i++) {

          auto eligible_col_id_start =
              (batch_id >= 0) ? batch_id * batch_size : 0;
          auto eligible_col_id_end =
              (batch_id >= 0)
                  ? std::min(static_cast<uint64_t>((batch_id + 1) * batch_size),
                             static_cast<uint64_t>(proc_col_width))
                  : proc_col_width;
          if (rank != r and
              (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1];
                 j++) {
              auto col_val = handle->col_idx[j];
              if (col_val >= eligible_col_id_start and
                  col_val < eligible_col_id_end) {
                // calculation of sender col_ids
                { proc_to_id_mapping[r].push_back(col_val); }

                if (alpha < 1.0)
                  count++;
              }
            }
          }
          if (alpha < 1.0 and count >= per_batch_nnz)
            break;
        }
        cout<<" rank "<<rank <<" sending calc  "<<count<<" to process "<<r<<" total nnz"<<total_nnz<<" effective "<<effective_nnz<<" for per batch nnz "<<per_batch_nnz<<endl;
      }
    } else if (transpose) {

      // calculation of receiver col_ids
//#pragma omp parallel for
      for (int r = 0; r < world_size; r++) {
        uint64_t starting_index =
            (batch_id >= 0) ? batch_id * batch_size + proc_col_width * r
                            : proc_col_width * r;
        auto end_index =
            (batch_id >= 0)
                ? std::min(
                      starting_index + batch_size,
                      std::min(static_cast<uint64_t>((r + 1) * proc_col_width),
                               gCols)) -
                      1
                : std::min(static_cast<uint64_t>((r + 1) * proc_col_width),
                           gCols) -
                      1;

        auto per_batch_nnz = 0;
        int count = 0;
        auto considered_range_start =
            (batch_id < batches - 1) ? (batch_id + 1) * batch_size : 0;
        if (0 < alpha < 1.0) {
          auto starting_index_co = proc_col_width * r;
          auto end_index_co =
              std::min(static_cast<uint64_t>((r + 1) * proc_col_width), gCols);
          auto total_nnz = handle->rowStart[end_index_co] -
                           handle->rowStart[starting_index_co];
          auto effective_nnz = alpha * total_nnz;
          per_batch_nnz = effective_nnz / batches;
        }

        for (auto i = starting_index; i <= (end_index); i++) {
          if (rank != r and
              (handle->rowStart[i + 1] - handle->rowStart[i]) > 0 and
              alpha == 1.0) {
            proc_to_id_mapping[r].push_back(i);
          } else if (rank != r and
                     (handle->rowStart[i + 1] - handle->rowStart[i]) > 0 and
                     alpha < 1.0) {
            for (int j = handle->rowStart[i]; j < handle->rowStart[i + 1];
                 j++) {
              if (considered_range_start >= handle->col_idx[j]) {
                proc_to_id_mapping[r].push_back(i);
                count++;
              }
            }

            if (count >= per_batch_nnz)
              break;
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

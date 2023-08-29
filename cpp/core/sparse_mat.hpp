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
  unique_ptr<CSRLocal<T>> csr_local_data;

public:
  uint64_t gRows, gCols, gNNz;
  vector<Tuple<T>> coords;
  int batch_size;
  int proc_col_width, proc_row_width;
  bool transpose = false;
  bool col_partitioned = false;

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

  CSRLinkedList<T> *get_batch_list(int batch_id) {
    return csr_linked_lists[batch_id].get();
  }

  void print_blocks_and_cols(bool trans) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //    cout << " rank " << rank << "_" << trans
    //         << " printing print_blocks_and_cols" << endl;
    int current_col_block = 0;
    for (int j = 0; j < csr_linked_lists.size(); j++) {
      if (j == 0 or
          j == csr_linked_lists.size() - 1) { // print first and last one
        cout << " rank " << rank << " j " << j << endl;
        auto linkedList = csr_linked_lists[j];

        auto head = (linkedList.get())->getHeadNode();

        int count = 0;
        while (head != nullptr) {
          string output_path = "blocks_rank" + to_string(rank) + "_trans" +
                               to_string(trans) + "_col_" +
                               to_string((trans) ? j : count) + "_row_" +
                               to_string((trans) ? count : j) + ".txt";
          char stats[500];
          strcpy(stats, output_path.c_str());
          ofstream fout(stats, std::ios_base::app);

          auto csr_data = (head.get())->data;

          int num_coords = (csr_data.get())->num_coords;

          cout << " rank " << rank << " j " << j << " num_coords " << num_coords
               << "_col_" + to_string((trans) ? j : count) + "_row_" +
                      to_string((trans) ? count : j)
               << endl;
          if (num_coords > 0) {
            distblas::core::CSRHandle *handle = (csr_data.get())->handler.get();
            int numRows = handle->rowStart.size() - 1;

            for (int i = 0; i < numRows; i++) {
              int start = handle->rowStart[i];
              int end = handle->rowStart[i + 1];
              fout << "Row " << i << ": ";
              if (num_coords > 0) {
                for (int k = start; k < end; k++) {

                  int col = handle->col_idx[k];
                  int value = handle->values[k];

                  if (value > 60000) {
                    cout << "Rank " << rank << " j " << j
                         << " Large value encountered "
                         << " Row " << i << " col " << col << " value " << value
                         << endl;
                  }
                  fout << "(" << col << ", " << value << ") ";
                }
              }
              fout << endl;
            }
          }
          head = (head.get())->next;
          ++count;
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

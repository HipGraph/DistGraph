#pragma once
#include "common.h"
#include "csr_linked_list.hpp"
#include "csr_local.hpp"
#include "distributed_mat.hpp"
#include <algorithm>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <parallel/algorithm>
#include <vector>

using namespace std;

namespace distblas::core {

/**
 * This class represents the Sparse Matrix
 */
template <typename T> class SpMat : public DistributedMat {

public:
  int gRows, gCols, gNNz;
  vector<Tuple<T>> coords;

  vector<uint64_t> block_col_starts;
  vector<uint64_t> block_row_starts;

  vector<shared_ptr<CSRLinkedList<T>>> csr_linked_lists;

  /**
   * Constructor for Sparse Matrix representation of  Adj matrix
   * @param coords  (src, dst, value) Tuple vector loaded as input
   * @param gRows   total number of Rows in Distributed global Adj matrix
   * @param gCols   total number of Cols in Distributed global Adj matrix
   * @param gNNz     total number of NNz in Distributed global Adj matrix
   */
  SpMat(vector<Tuple<T>> &coords, int gRows, int gCols, int gNNz) {
    this->gRows = gRows;
    this->gCols = gCols;
    this->gNNz = gNNz;
    this->coords = coords;
  }

  SpMat() {}

  void divide_block_cols(int block_width, int target_divisions, bool mod_ind,
                         bool trans) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    block_col_starts.clear();

    // Locate block starts within the local sparse matrix (i.e. divide a long
    // block row into subtiles)
    int current_start = 0;
    if (trans) {
      current_start = block_width * rank;
    }

    cout << "rank " << rank << " trans" << trans << " current_start "
         << current_start << endl;

    // TODO: introduce atomic capture
    for (uint64_t i = 0; i < coords.size(); i++) {
      while (coords[i].col >= current_start) {
        block_col_starts.push_back(i);
        cout << "rank " << rank << " trans" << trans << " col adding i " << i
             << endl;
        current_start += block_width;
      }

      // This modding step helps indexing.
      if (mod_ind) {
        coords[i].col %= block_width;
      }
    }

    assert(block_col_starts.size() <= target_divisions + 1);

    while (block_col_starts.size() < target_divisions + 1) {
      cout << "rank " << rank << " trans" << trans << " col adding i "
           << coords.size() << endl;
      block_col_starts.push_back(coords.size());
    }
  }

  void sort_by_rows() {
    for (int i = 0; i < block_col_starts.size() - 1; i++) {
      __gnu_parallel::sort(coords.begin() + block_col_starts[i],
                           coords.begin() + block_col_starts[i + 1],
                           row_major<T>);
    }
  }

  void divide_block_rows(int block_width_row, int block_width_col,
                         int target_divisions, bool mod_ind, bool trans) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    block_row_starts.clear();
    for (uint64_t i = 0; i < block_col_starts.size() - 1; i++) {

      int current_start = block_width_col * rank;

      if (trans) {
        current_start = 0;
      }

      // TODO: introduce atomic capture
      for (uint64_t j = block_col_starts[i]; j < block_col_starts[i + 1]; j++) {
        while (coords[j].row >= current_start) {
          block_row_starts.push_back(j);
          cout << "rank " << rank << " trans" << trans << " row adding j " << j
               << endl;
          current_start += block_width_row;
        }

        // This modding step helps indexing.
        if (mod_ind) {
          coords[j].row %= block_width_row;
        }
      }
    }
  }

  void initialize_CSR_blocks(int block_rows, int block_cols, int max_nnz,
                             bool transpose) {

    int current_block = 0;
    int current_vector_pos = 0;

    vector<uint64_t> operator_vec =
        (transpose) ? block_col_starts : block_row_starts;
    vector<uint64_t> operator_vec_comp =
        (transpose) ? block_row_starts : block_col_starts;

    csr_linked_lists = std::vector<std::shared_ptr<CSRLinkedList<T>>>(
        operator_vec.size() - 1, std::make_shared<CSRLinkedList<T>>());

    for (int j = 0; j < operator_vec.size() - 1; j++) {

      if (operator_vec[j] >= operator_vec_comp[current_block + 1]) {
        ++current_block;
        current_vector_pos = 0;
      }

      int num_coords = operator_vec[j + 1] - operator_vec[j];

      if (num_coords > 0) {
        Tuple<T> *coords_ptr = (coords.data() + operator_vec[j]);
        (csr_linked_lists[current_vector_pos].get())
            ->insert(block_rows, block_cols, num_coords, coords_ptr, num_coords,
                     transpose);
        ++current_vector_pos;
      }
    }
  }

  void print_blocks_and_cols(bool trans) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cout << " rank " << rank << "_" << trans
         << " printing print_blocks_and_cols" << endl;
    int current_col_block = 0;
    for (int j = 0; j < csr_linked_lists.size(); j++) {
      cout << " rank " << rank << " j " << j << endl;
      auto linkedList = csr_linked_lists[j];

      auto head = (linkedList.get())->getHeadNode();

      int count = 0;
      while (head != nullptr) {
        string output_path = "blocks_rank" + to_string(rank) + "_trans" +
                             to_string(trans) + "_col_" + to_string(count) +
                             "_row_" + to_string(j) + ".txt";
        char stats[500];
        strcpy(stats, output_path.c_str());
        ofstream fout(stats, std::ios_base::app);

        auto csr_data = (head.get())->data;

        distblas::core::CSRHandle *handle = (csr_data.get())->handler.get();
        int numRows = handle->rowStart.size() - 1;

        for (int i = 0; i < numRows; i++) {
          int start = handle->rowStart[i];
          int end = handle->rowStart[i + 1];

          fout << "Row " << i << ": ";
          for (int k = start; k < end; k++) {
            int col = handle->col_idx[k];
            int value = handle->values[k];

            fout << "(" << col << ", " << value << ") ";
          }
          fout << endl;
        }
        head = (head.get())->next;
        ++count;
      }
    }
  }

  void print_coords(bool trans) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    string output_path = "coords" + to_string(rank) ".txt";
    char stats[500];
    strcpy(stats, output_path.c_str());
    ofstream fout(stats, std::ios_base::app);

    for (int i = 0; i < coords.size(); i++) {
      fout << coords[i].row << " " << coords[i].col << " " << endl;
    }
  }

  ~SpMat() {}
};

} // namespace distblas::core

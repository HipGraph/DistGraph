#pragma once
#include "../core/dense_mat.hpp"
#include "../core/sparse_mat.hpp"
#include "../net/process_3D_grid.hpp"
#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <string>
#include <vector>

using namespace std;
using namespace distblas::core;
using namespace combblas;
namespace distblas::io {

// typedef SpParMat<int64_t , double , SpDCCols<int64_t, double>>
// PSpMat_s32p64_Int;

/**
 * This class implements IO operations of DistBlas library.
 */
class ParallelIO {
private:
public:
  ParallelIO();
  ~ParallelIO();

  /**
   * Interface for parallel reading of Matrix Market formatted files
   * @param file_path
   */
  template <typename INDEX_TYPE, typename WEIGHT_VALUE, typename VALUE_TYPE>
  void parallel_read_MM(string file_path,
                        distblas::core::SpMat<VALUE_TYPE> *sp_mat,
                        bool copy_col_to_value) {
    MPI_Comm WORLD;
    MPI_Comm_dup(MPI_COMM_WORLD, &WORLD);

    int proc_rank, num_procs;
    MPI_Comm_rank(WORLD, &proc_rank);
    MPI_Comm_size(WORLD, &num_procs);

    shared_ptr<CommGrid> simpleGrid;
    simpleGrid.reset(new CommGrid(WORLD, num_procs, 1));

    SpParMat<INDEX_TYPE, WEIGHT_VALUE, SpDCCols<INDEX_TYPE, WEIGHT_VALUE>> G(
        simpleGrid);
    //    unique_ptr<PSpMat_s32p64_Int> G =
    //        unique_ptr<PSpMat_s32p64_Int>(new PSpMat_s32p64_Int(simpleGrid));

    INDEX_TYPE nnz;

    G.ParallelReadMM(file_path, true, maximum<WEIGHT_VALUE>());

    nnz = G.getnnz();
    if (proc_rank == 0) {
      cout << "File reader read " << nnz << " nonzeros." << endl;
    }
    SpTuples<int64_t, WEIGHT_VALUE> tups(G.seq());
    tuple<int64_t, int64_t, WEIGHT_VALUE> *values = tups.tuples;

    vector<Tuple<VALUE_TYPE>> coords;
    coords.resize(tups.getnnz());

#pragma omp parallel for
    for (int i = 0; i < tups.getnnz(); i++) {
      coords[i].row = get<0>(values[i]);
      coords[i].col = get<1>(values[i]);
      if (copy_col_to_value) {
        coords[i].value = get<1>(values[i]);
      } else {
        coords[i].value = get<2>(values[i]);
      }
    }

    int rowIncrement = G.getnrow() / num_procs;

#pragma omp parallel for
    for (int i = 0; i < coords.size(); i++) {
      coords[i].row += rowIncrement * proc_rank;
    }

    sp_mat->coords = coords;
    sp_mat->gRows = G.getnrow();
    sp_mat->gCols = G.getncol();
    sp_mat->gNNz = G.getnnz();
  }

  template <typename VALUE_TYPE>
  void parallel_write(string file_path, VALUE_TYPE *nCoordinates,
                      INDEX_TYPE rows, uint64_t cols, Process3DGrid *grid,
                      distblas::core::SpMat<VALUE_TYPE> *sp_mat) {
    MPI_File fh;
    MPI_File_open(grid->col_world, file_path.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    uint64_t expected_rows = rows;
    if (grid->rank_in_col == grid->col_world_size - 1) {
      auto expected_last_rows = sp_mat->gRows - rows * grid->rank_in_col;
      cout << " expected rows " << expected_last_rows << endl;
      expected_rows = min(expected_last_rows, rows);
    }

    //    int offset= rows-expected_rows;
    cout << " rank :" << grid->rank_in_col << " expected rows " << expected_rows
         << endl;
    size_t total_size = 0;
    for (INDEX_TYPE i = 0; i < expected_rows; ++i) {
      total_size +=
          snprintf(nullptr, 0, "%lu", i + 1 + grid->rank_in_col * rows);
      for (int j = 0; j < cols; ++j) {
        total_size += snprintf(nullptr, 0, " %.5f", nCoordinates[i * cols + j]);
      }
      total_size += snprintf(nullptr, 0, "\n");
    }

    // Allocate memory dynamically
    char *buffer =
        (char *)malloc(total_size + 1); // +1 for the null-terminating character
    if (buffer == nullptr) {
      // Handle allocation failure
      cout << "Memory allocation failed." << endl;
      return;
    }

    char *current_position = buffer;

    for (INDEX_TYPE i = 0; i < expected_rows; ++i) {
      current_position += snprintf(current_position, total_size, "%lu",
                                   i + 1 + grid->rank_in_col * rows);
      for (int j = 0; j < cols; ++j) {
        current_position += snprintf(current_position, total_size, " %.5f",
                                     nCoordinates[i * cols + j]);
      }
      current_position += snprintf(current_position, total_size, "\n");
    }

    MPI_File_write_ordered(fh, buffer, current_position - buffer, MPI_CHAR,
                           MPI_STATUS_IGNORE);

    // Free the dynamically allocated memory
    free(buffer);

    MPI_File_close(&fh);
  }

  template <typename VALUE_TYPE>
  void build_sparse_random_matrix(INDEX_TYPE rows,INDEX_TYPE global_rows, INDEX_TYPE cols,INDEX_TYPE global_cols,
                                  double density, int seed,
                                  vector<Tuple<VALUE_TYPE>> &sparse_coo,
                                  Process3DGrid *grid, bool bfs_input = false) {
    std::mt19937 gen(seed);
    std::mt19937 gen1(seed+1);
    std::normal_distribution<VALUE_TYPE> norm_dist(0, 1);

    if (bfs_input) {
      INDEX_TYPE start_index = grid->rank_in_col*rows;

      INDEX_TYPE start_col_index = grid->rank_in_col*cols;
      INDEX_TYPE end_col_index = min((grid->rank_in_col+1)*cols,global_cols);

      std::uniform_int_distribution<INDEX_TYPE> uni_dist(start_col_index, end_col_index - 1);
      std::uniform_int_distribution<INDEX_TYPE> uni_dist_rows(0, rows - 1);
      std::unordered_set<INDEX_TYPE> indexes_taken;
      std::unordered_set<INDEX_TYPE> rows_taken;
      INDEX_TYPE row;
      int count = 0;
      while (count < cols) {
        row = static_cast<INDEX_TYPE>(uni_dist_rows(gen));
        row = min(start_index +row,global_rows);
        if (rows_taken.insert(row).second) {
          INDEX_TYPE index = static_cast<INDEX_TYPE>(uni_dist(gen1));
          if (indexes_taken.insert(index).second) {
            Tuple<VALUE_TYPE> t;
            cout<<" rank "<<grid->rank_in_col<<" row "<<row<<" index "<<index<<endl;
            t.row = row;   // Calculate row index
            t.col = index ; // Calculate column index
            t.value = 1;
            sparse_coo.push_back(t);
            count++;
          }
        }
      }
    } else {
      auto expected_non_zeros = cols * density;
      std::uniform_real_distribution<VALUE_TYPE> uni_dist(0, cols - 1);
      for (INDEX_TYPE i = 0; i < rows; ++i) {
        for (INDEX_TYPE j = 0; j < expected_non_zeros; j++) {
          VALUE_TYPE val = static_cast<VALUE_TYPE>(norm_dist(gen));
          auto index = uni_dist(gen);
          Tuple<VALUE_TYPE> t;
          t.row = i;     // Calculate row index
          t.col = index; // Calculate column index
          t.value = val;
          sparse_coo.push_back(t);
        }
      }
    }
  }

  template <typename VALUE_TYPE>
  void parallel_write(string file_path, vector<Tuple<VALUE_TYPE>> &sparse_coo,
                      Process3DGrid *grid, INDEX_TYPE local_rows,
                      INDEX_TYPE global_rows, INDEX_TYPE global_cols) {
    MPI_File fh;
    MPI_File_open(grid->col_world, file_path.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    int chunk_size = 100000; // Number of elements to write at a time
    size_t total_size = 0;

    INDEX_TYPE global_sum = 0;
    INDEX_TYPE local_sum = sparse_coo.size();

    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_UINT64_T, MPI_SUM,grid->col_world);

    int increment = min(chunk_size, static_cast<int>(sparse_coo.size()));

    for (INDEX_TYPE i = 0; i < sparse_coo.size(); i += increment) {
      if (grid->rank_in_col == 0 and i == 0) {
        total_size += snprintf(
            nullptr, 0,
            "%%%MatrixMarket matrix coordinate real general\n%lu %lu %lu\n",
            global_rows, global_cols, global_sum);
      }
      int elements_in_chunk = min(chunk_size, static_cast<int>(sparse_coo.size() - i));

      for (int j = 0; j < elements_in_chunk; ++j) {
        Tuple<VALUE_TYPE> t = sparse_coo[i + j];
        INDEX_TYPE row = static_cast<INDEX_TYPE>(t.row + 1);
        INDEX_TYPE col = static_cast<INDEX_TYPE>(t.col + 1);
        total_size += snprintf(nullptr, 0, "%lu %lu %.5f\n", row, col, t.value);
      }

      char *buffer = (char *)malloc(total_size +
                                    1); // +1 for the null-terminating character
      if (buffer == nullptr) {
        // Handle allocation failure
        cout << "Memory allocation failed." << endl;
        return;
      }

      char *current_position = buffer;
      if (i == 0 and grid->rank_in_col == 0) {
        current_position += snprintf(
            current_position, total_size,
            "%%%MatrixMarket matrix coordinate real general\n%lu %lu %lu\n",
            global_rows, global_cols, global_sum);
      }

      for (int j = 0; j < elements_in_chunk; ++j) {
        Tuple<VALUE_TYPE> t = sparse_coo[i + j];
        INDEX_TYPE row = static_cast<INDEX_TYPE>(t.row + 1);
        INDEX_TYPE col = static_cast<INDEX_TYPE>(t.col + 1);
        current_position += snprintf(current_position, total_size,
                                     "%lu %lu %.5f\n", row, col, t.value);
      }

      MPI_Status status;
      MPI_File_write_ordered(fh, buffer, current_position - buffer, MPI_CHAR,
                             MPI_STATUS_IGNORE);

      // Free the dynamically allocated memory for each chunk
      free(buffer);
      total_size = 0; // Reset total_size for the next chunk
    }

    // Ensure that all processes have completed their writes
    MPI_File_close(&fh);
  }

  template <typename VALUE_TYPE>
  void parallel_write_csr(string file_path, CSRHandle *handle,
                          Process3DGrid *grid, INDEX_TYPE local_rows,
                          INDEX_TYPE global_rows, INDEX_TYPE global_cols) {
    MPI_File fh;
    MPI_File_open(grid->col_world, file_path.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    int chunk_size = 100000; // Number of elements to write at a time
    size_t total_size = 0;

    INDEX_TYPE global_sum = 0;
    INDEX_TYPE local_sum = (*handle).rowStart[(*handle).rowStart.size() - 1];

    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_UINT64_T, MPI_SUM,
                  grid->col_world);

    if (grid->rank_in_col == 0) {
      total_size += snprintf(
          nullptr, 0,
          "%%%MatrixMarket matrix coordinate real general\n%lu %lu %lu\n",
          global_rows, global_cols, global_sum);
    }

    for (auto i = 0; i < (*handle).rowStart.size(); i++) {
      for (auto j = (*handle).rowStart[i]; j < (*handle).rowStart[i + 1]; j++) {
        INDEX_TYPE index = i + grid->rank_in_col * local_rows;
        INDEX_TYPE col_index = (*handle).col_idx[j];
        VALUE_TYPE value = (*handle).values[j];
        total_size +=
            snprintf(nullptr, 0, "%lu %lu %.5f\n", index, col_index, value);
      }
      char *buffer = (char *)malloc(total_size +
                                    1); // +1 for the null-terminating character
      if (buffer == nullptr) {
        // Handle allocation failure
        cout << "Memory allocation failed." << endl;
        return;
      }

      char *current_position = buffer;
      if (i == 0 and grid->rank_in_col == 0) {
        current_position += snprintf(
            current_position, total_size,
            "%%%MatrixMarket matrix coordinate real general\n%lu %lu %lu\n",
            global_rows, global_cols, global_sum);
      }
      for (auto j = (*handle).rowStart[i]; j < (*handle).rowStart[i + 1]; j++) {
        INDEX_TYPE index = i + grid->rank_in_col * local_rows;
        INDEX_TYPE col_index = (*handle).col_idx[j];
        VALUE_TYPE value = (*handle).values[j];
        current_position += snprintf(current_position, total_size,
                                     "%lu %lu %.5f\n", index, col_index, value);
      }
      MPI_Status status;
      MPI_File_write_ordered(fh, buffer, current_position - buffer, MPI_CHAR,
                             MPI_STATUS_IGNORE);

      // Free the dynamically allocated memory for each chunk
      free(buffer);
      total_size = 0; // Reset total_size for the next chunk
    }

    MPI_File_close(&fh);
  }
};
} // namespace distblas::io

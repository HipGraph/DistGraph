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

typedef SpParMat<int64_t, int, SpDCCols<int32_t, int>> PSpMat_s32p64_Int;

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
  template <typename T>
  void parallel_read_MM(string file_path, distblas::core::SpMat<T> *sp_mat,
                        bool copy_col_to_value) {
    MPI_Comm WORLD;
    MPI_Comm_dup(MPI_COMM_WORLD, &WORLD);

    int proc_rank, num_procs;
    MPI_Comm_rank(WORLD, &proc_rank);
    MPI_Comm_size(WORLD, &num_procs);

    shared_ptr<CommGrid> simpleGrid;
    simpleGrid.reset(new CommGrid(WORLD, num_procs, 1));

    unique_ptr<PSpMat_s32p64_Int> G =
        unique_ptr<PSpMat_s32p64_Int>(new PSpMat_s32p64_Int(simpleGrid));

    uint64_t nnz;

    G.get()->ParallelReadMM(file_path, true, maximum<T>());

    nnz = G.get()->getnnz();
    if (proc_rank == 0) {
      cout << "File reader read " << nnz << " nonzeros." << endl;
    }
    SpTuples<int64_t, T> tups(G.get()->seq());
    tuple<int64_t, int64_t, T> *values = tups.tuples;

    vector<Tuple<T>> coords;
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

    int rowIncrement = G->getnrow() / num_procs;

#pragma omp parallel for
    for (int i = 0; i < coords.size(); i++) {
      coords[i].row += rowIncrement * proc_rank;
    }

    sp_mat->coords = coords;
    sp_mat->gRows = G.get()->getnrow();
    sp_mat->gCols = G.get()->getncol();
    sp_mat->gNNz = G.get()->getnnz();
  }

  template <typename T, typename SPT>
  void parallel_write(string file_path, T *nCoordinates, uint64_t rows,
                      uint64_t cols, Process3DGrid *grid,
                      distblas::core::SpMat<SPT> *sp_mat) {
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
    for (uint64_t i = 0; i < expected_rows; ++i) {
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

    for (uint64_t i = 0; i < expected_rows; ++i) {
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

  template <typename T>
  void build_sparse_random_matrix(int rows, int cols, double density, int seed,
                                  vector<Tuple<T>> &sparse_coo,Process3DGrid *grid) {

    std::mt19937 gen(seed);
//    std::uniform_real_distribution<double> uni_dist(0, 1);
    std::normal_distribution<double> norm_dist(0, 1);

    int expected_non_zeros  = cols*density;

//    // follow row major order
//    #pragma omp parallel for
//    for (int i = 0; i < rows * cols; ++i) {
//      if (uni_dist(gen) <= density) {
//        T val = static_cast<T>(norm_dist(gen));
//        Tuple<T> t;
//        t.row = i / cols;  // Calculate row index
//        t.col = i % cols;  // Calculate column index
//        t.value = val;
//
//        #pragma omp critical
//        sparse_coo.push_back(t);
//      }
//    }
    std::uniform_real_distribution<double> uni_dist(0, cols);
        for (int i = 0; i < rows; ++i) {
          for(int j=0;j<expected_non_zeros;j++){
            T val = static_cast<T>(norm_dist(gen));
            int index = uni_dist(gen);
            Tuple<T> t;
            t.row = i ;  // Calculate row index
            t.col = index;  // Calculate column index
            t.value = val;
            sparse_coo.push_back(t);
          }
        }
  }


  template <typename T>
  void parallel_write(string file_path, vector<Tuple<T>> &sparse_coo, Process3DGrid *grid, int rows, uint64_t global_rows, uint64_t global_cols) {
    MPI_File fh;
    MPI_File_open(grid->col_world, file_path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    int chunk_size = 100000; // Number of elements to write at a time
    size_t total_size = 0;

    uint64_t  global_sum = 0;
    uint64_t local_sum = sparse_coo.size();

    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_UINT64_T, MPI_SUM, grid->col_world);


    for (uint64_t i = 0; i < sparse_coo.size(); i += chunk_size) {
      if (grid->rank_in_col==0 and i==0){
        total_size += snprintf(nullptr, 0, "%%%MatrixMarket matrix coordinate real general\n%lu %lu %lu\n", global_rows, global_cols, global_sum);
      }
      int elements_in_chunk = min(chunk_size, static_cast<int>(sparse_coo.size() - i));

      for (int j = 0; j < elements_in_chunk; ++j) {
        Tuple<T> t = sparse_coo[i + j];
        int col = static_cast<int>(t.col + 1);
        uint64_t row = static_cast<uint64_t>(t.row + 1 + grid->rank_in_col * rows);
        total_size += snprintf(nullptr, 0, "%lu %lu %.5f\n", row, col, t.value);
      }

      char *buffer = (char *)malloc(total_size + 1); // +1 for the null-terminating character
      if (buffer == nullptr) {
        // Handle allocation failure
        cout << "Memory allocation failed." << endl;
        return;
      }

      char *current_position = buffer;
      if (i==0 and grid->rank_in_col==0){
        current_position += snprintf(current_position, total_size, "%%%MatrixMarket matrix coordinate real general\n%lu %lu %lu\n", global_rows, global_rows, global_sum);
      }

      for (int j = 0; j < elements_in_chunk; ++j) {
        Tuple<T> t = sparse_coo[i + j];
        uint64_t col = static_cast<int>(t.col + 1);
        uint64_t row = static_cast<uint64_t>(t.row + 1 + grid->rank_in_col * rows);
        current_position += snprintf(current_position, total_size, "%lu %lu %.5f\n", row, col, t.value);
      }

      MPI_Status status;
      MPI_File_write_ordered(fh, buffer, current_position - buffer, MPI_CHAR, MPI_STATUS_IGNORE);

      // Free the dynamically allocated memory for each chunk
      free(buffer);
      total_size = 0; // Reset total_size for the next chunk
    }

    // Ensure that all processes have completed their writes
    MPI_File_close(&fh);
  }

};
} // namespace distblas::io

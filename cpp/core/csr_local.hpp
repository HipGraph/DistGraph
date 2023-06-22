#pragma once
#include "common.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <mkl_spblas.h>
#include <mpi.h>
#include <numeric>
#include <parallel/algorithm>
#include <string.h>
#include <cassert>
#include <fstream>
#include <mkl.h>

using namespace std;

namespace distblas::core {

template <typename T>
class CSRLocal {

public:
  MKL_INT rows, cols;

  int max_nnz, num_coords;

  bool transpose;

  int active;

  CSRHandle *buffer;



  CSRLocal() {}
  /*
   * TODO: Need to check this function for memory leaks!
   */
  CSRLocal(MKL_INT rows, MKL_INT cols, MKL_INT max_nnz, Tuple<T> *coords,
           int num_coords, bool transpose ) {
    this->transpose = transpose;
    this->num_coords = num_coords;
    this->rows = rows;
    this->cols = cols;

    this->buffer = new CSRHandle[2];

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // This setup is really clunky, but I don't have time to fix it.
    vector<MKL_INT> rArray(1000, 0);
    vector<MKL_INT> cArray(1000, 0);
    vector<double> vArray(1000, 0.0);
//
//    // Put a dummy value in if the number of coordinates is 0, so that
//    // everything doesn't blow up
//    if (num_coords == 0) {
//      rArray.push_back(0);
//      cArray.push_back(0);
//      vArray.push_back(0.0);
//    }



//#pragma omp parallel for
    for (int i = 0; i < 5000; i++) {
//      rArray[i] = coords[i].row;
//      cArray[i] = coords[i].col;
      rArray[i] = i;
      cArray[i] = i;
//      vArray[i] = static_cast<double>(coords[i].value);
      vArray[i] = 1.0;
    }

//    string output_path =  "output.txt"+ to_string(rank);
//    char stats[500];
//    strcpy(stats, output_path.c_str());
//    ofstream fout(stats, std::ios_base::app);
//
//    for (int i = 0; i < num_coords; i++) {
//
//      fout<<" rank "<<rank<<" "<< rArray[i] << " "<< cArray[i]<<" " << vArray[i] <<endl;
//    }

//    mkl_set_num_threads(1);

//    std::vector<MKL_INT> rArray = {0, 1, 2};
//    std::vector<MKL_INT> cArray = {0, 1, 2};
//    std::vector<double> vArray = {1.0, 2.0, 3.0};

    sparse_operation_t op;
    if (transpose) {
      op = SPARSE_OPERATION_TRANSPOSE;
    } else {
      op = SPARSE_OPERATION_NON_TRANSPOSE;
    }

    sparse_matrix_t tempCOO, tempCSR;

    cout<<" rank "<< rank <<" number of coords "<< num_coords << " attempting to create coo "<<endl;

    sparse_status_t  status_coo = mkl_sparse_d_create_coo(&tempCOO, SPARSE_INDEX_BASE_ZERO, 30000, 30000,
                            max(5000, 1), rArray.data(), cArray.data(),
                            vArray.data());

    if (status_coo != SPARSE_STATUS_SUCCESS) {
      std::cerr << " rank "<<rank<< "Error: Conversion from COO to CSR failed." << std::endl;
      // Handle the error or exit the program
    }

    cout<<" rank "<< rank << " mkl_sparse_d_create_coo  stats: " << status_coo << endl;

    sparse_status_t  status_csr = mkl_sparse_convert_csr(tempCOO, op, &tempCSR);

    cout<<" rank "<< rank << " mkl_sparse_convert_csr  stats: " << status_csr <<endl;

    mkl_sparse_destroy(tempCOO);


    cout<<" rank "<< rank <<" number of coords "<< num_coords << "  coo destroy succeeded "<<endl;


    vector<MKL_INT>().swap(rArray);
    vector<MKL_INT>().swap(cArray);
    vector<double>().swap(vArray);

    sparse_index_base_t indexing;
    MKL_INT *rows_start, *rows_end, *col_idx;
    double *values;


    cout<<" rank "<< rank <<" number of coords "<< num_coords << "  csr exported "<<endl;

    mkl_sparse_d_export_csr(tempCSR, &indexing, &(this->rows), &(this->cols),
                            &rows_start, &rows_end, &col_idx, &values);


    cout<<" rank "<< rank <<" number of coords "<< num_coords << "  csr exported done "<<endl;

    int rv = 0;
    for (int i = 0; i < num_coords; i++) {
      while (rv < this->rows && i >= rows_start[rv + 1]) {
        rv++;
      }
      coords[i].row = rv;
      coords[i].col = col_idx[i];
      coords[i].value = static_cast<T>(values[i]);
    }

    active = 0;

    assert(num_coords <= max_nnz);

    for (int t = 0; t < 2; t++) {
      buffer[t].values.resize(max_nnz == 0 ? 1 : max_nnz);
      buffer[t].col_idx.resize(max_nnz == 0 ? 1 : max_nnz);
      buffer[t].row_idx.resize(max_nnz == 0 ? 1 : max_nnz);
      buffer[t].rowStart.resize(this->rows + 1);

// Copy over row indices
#pragma omp parallel for
      for (int i = 0; i < num_coords; i++) {
        buffer[t].row_idx[i] = coords[i].row;
      }

      memcpy(buffer[t].values.data(), values,
             sizeof(double) * max(num_coords, 1));
      memcpy(buffer[t].col_idx.data(), col_idx,
             sizeof(MKL_INT) * max(num_coords, 1));
      memcpy(buffer[t].rowStart.data(), rows_start,
             sizeof(MKL_INT) * this->rows);

      buffer[t].rowStart[this->rows] = max(num_coords, 1);


      cout<<" rank "<< rank <<" number of coords "<< num_coords << " creating csr ... "<<endl;

      mkl_sparse_d_create_csr(
          &(buffer[t].mkl_handle), SPARSE_INDEX_BASE_ZERO, this->rows,
          this->cols, buffer[t].rowStart.data(), buffer[t].rowStart.data() + 1,
          buffer[t].col_idx.data(), buffer[t].values.data());


      cout<<" rank "<< rank <<" number of coords "<< num_coords << " creating csr completed "<<endl;

      // This madness is just trying to get around the inspector routine
      if (num_coords == 0) {
        buffer[t].rowStart[this->rows] = 0;
      }
    }

    mkl_sparse_destroy(tempCSR);
  }

  ~CSRLocal() {
    for (int t = 0; t < 2; t++) {
      mkl_sparse_destroy(buffer[t].mkl_handle);
    }
    delete[] buffer;
  }
};

} // namespace distblas::core

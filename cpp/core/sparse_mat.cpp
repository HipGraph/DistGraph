#include "sparse_mat.hpp"

using namespace distblas::core;

template <typename T>
SpMat<T>::SpMat(vector<Tuple<T>> &coords, int gRows, int gCols, int gNNz) {
  this->gRows = gRows;
  this->gCols = gCols;
  this->gNNz = gNNz;
  this->coords = coords;
}

template <typename T>
SpMat<T>::~SpMat() {

}

template <typename T>
void SpMat<T>::initialize_csr() {

}

#include "../include/DistBLAS/SpMat.hpp"


using namespace distblas::core;

template <typename T>
SpMat::SpMat(vector<Tuple<T>> &coords, int gRows, int gCols, int gNNz) {
  this->gRows = gRows;
  this->gCols = gCols;
  this->gNNz = gNNz;
  this->coords = coords;
}

template <typename T>
SpMat::~SpMat() {

}

template <typename T>
void SpMat::initialize_csr() {

}

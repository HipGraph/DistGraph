#include "../partition/partitioner.hpp"
using namespace distblas::partition;
using namespace std;

Partitioner::Partitioner(MPI_Comm comm) {
 this->comm = comm;
}

Partitioner::~Partitioner() {

}




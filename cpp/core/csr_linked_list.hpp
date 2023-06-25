#pragma once
#include "common.h"
#include "csr_local.hpp"

using namespace std;

namespace distblas::core {

template <typename T>
struct CSRLocalNode {
  CSRLocal<T>* data;
  CSRLocalNode<T> *next;
};

template <typename T>
class CSRLinkedList {

private:
  CSRLocalNode<T>* head;

public:

  CSRLinkedList() {
    head = nullptr;
  }

  ~CSRLinkedList() {
   CSRLocalNode<T>* temp = head;
   while(temp != nullptr) {
     CSRLocalNode<T>* nextTemp = temp->next;
     delete temp;
     temp = nextTemp;
   }
  }

  void insert(CSRLocal<T>* dataPoint) {
    CSRLocalNode<T>* newNode = new CSRLocalNode<T>();
    newNode->data = dataPoint;
    if (this->head == nullptr) {
      head = newNode;
    }else {
      CSRLocalNode<T>* temp = head;
      while(temp->next != nullptr) {
        temp = temp->next;
      }
      temp->next = newNode;
    }
  }

  CSRLocalNode<T>* getHeadNode() {
    return head;
  }

};

} // namespace distblas::core

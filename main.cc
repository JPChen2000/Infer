#include <iostream>
#include <memory>
#include "util/types.h"
#include "core/memory.h"
#include "core/tensor.h"
#include "core/dim.h"
#include "util/logger.h"

using namespace feather;
    

size_t size = 2 * 1024;
void new_delete() {
    auto buffer = Buffer(size);
}

int main() {


    std::shared_ptr<BufferPool> buffer_pool = BufferPool::getInstance();
    auto buffer = buffer_pool->allocate(1024); 
    auto tensor = Tensor(buffer); 
    std::vector<float> data = {0, 1, 2, 4, 5, 6, };
    std::vector<int64_t> shape = {2, 3}; 
    DDim dim(shape);

    tensor.Assign<float, DDim>(data.data(), dim);
     
    std::cout << tensor;
    return 0;
}

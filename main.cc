// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

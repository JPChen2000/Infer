#include "model/weight_store.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <limits>

namespace feather {
namespace model {

MappedFile::~MappedFile() { Close(); }

bool MappedFile::Open(const std::string& path) {
    Close();

    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        return false;
    }

    struct stat st {};
    if (fstat(fd_, &st) != 0 || st.st_size < 0) {
        Close();
        return false;
    }

    size_ = static_cast<size_t>(st.st_size);
    if (size_ == 0) {
        Close();
        return false;
    }

    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        Close();
        return false;
    }

    path_ = path;
    return true;
}

void MappedFile::Close() {
    if (data_ != nullptr) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    size_ = 0;
    path_.clear();
}

bool WeightStore::AddShard(const std::string& path) { return GetOrOpenShard(path) != nullptr; }

std::shared_ptr<Tensor> WeightStore::CreateTensorView(const TensorDesc& desc, const WeightLocation& location) {
    auto shard = GetOrOpenShard(location.shard_path);
    if (shard == nullptr) {
        return nullptr;
    }
    if (location.offset > std::numeric_limits<size_t>::max() ||
        location.byte_size > std::numeric_limits<size_t>::max()) {
        return nullptr;
    }
    const auto offset = static_cast<size_t>(location.offset);
    const auto byte_size = static_cast<size_t>(location.byte_size);
    if (offset > shard->size() || byte_size > shard->size() - offset) {
        return nullptr;
    }

    auto buffer = std::make_shared<Buffer>(shard->data(), shard->size(), shard);
    auto tensor = std::make_shared<Tensor>();
    tensor->Resize(desc.dims);
    tensor->set_data_type(desc.data_type);
    tensor->set_layout(desc.layout);
    tensor->set_quantization(desc.quantization);
    tensor->ResetBuffer(buffer, byte_size, offset);
    return tensor;
}

std::shared_ptr<MappedFile> WeightStore::GetOrOpenShard(const std::string& path) {
    auto it = shards_.find(path);
    if (it != shards_.end()) {
        return it->second;
    }

    auto shard = std::make_shared<MappedFile>();
    if (!shard->Open(path)) {
        return nullptr;
    }
    shards_[path] = shard;
    return shard;
}

}  // namespace model
}  // namespace feather

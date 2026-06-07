# YOLOv5 FTH Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable demo that loads a `.fth` YOLOv5 model, reads an image file, runs inference, and prints decoded detections after NMS.

**Architecture:** Keep the core inference framework unchanged and place demo-specific logic in isolated helper modules. The runtime path remains `ModelLoader -> StaticGraph -> GraphLowering -> RuntimeGraph`, while image decode, letterbox preprocessing, and YOLOv5 decode/NMS live in separate demo files.

**Tech Stack:** C++17, existing `infer` runtime, `gtest`, `stb_image` for image decoding

---

### Task 1: Scaffold the reusable demo helpers

**Files:**
- Create: `include/demo/image_io.h`
- Create: `include/demo/yolov5_postprocess.h`
- Create: `include/demo/yolov5_runner.h`
- Create: `src/demo/image_io.cc`
- Create: `src/demo/yolov5_postprocess.cc`
- Create: `src/demo/yolov5_runner.cc`

- [ ] **Step 1: Declare the helper interfaces**

```cpp
struct ImageData {
    int width{};
    int height{};
    int channels{};
    std::vector<uint8_t> pixels;
};

struct LetterboxInfo {
    int resized_width{};
    int resized_height{};
    int pad_x{};
    int pad_y{};
    float scale{};
};

struct Detection {
    int class_id{};
    float score{};
    float x1{};
    float y1{};
    float x2{};
    float y2{};
};

class Yolov5Runner {
   public:
    int32_t Load(const std::string& model_path);
    int32_t Run(const std::string& image_path, float conf_thresh, float iou_thresh,
                std::vector<Detection>* detections);
};
```

- [ ] **Step 2: Implement helper modules with no CLI code**

```cpp
int32_t LoadImage(const std::string& path, ImageData* image);
int32_t PreprocessImageToTensor(const ImageData& image, int input_size, feather::DataType dtype,
                                std::shared_ptr<feather::Tensor>* tensor, LetterboxInfo* letterbox);
std::vector<Detection> DecodeYolov5Detections(const feather::Tensor& output, const LetterboxInfo& letterbox,
                                              int image_width, int image_height,
                                              float conf_thresh, float iou_thresh);
```

- [ ] **Step 3: Keep runner ownership local to the demo module**

```cpp
class Yolov5Runner {
   private:
    feather::model::ModelLoader loader_;
    feather::StaticGraph static_graph_;
    feather::RuntimeGraph runtime_graph_;
    std::string input_name_;
    std::string output_name_;
    int input_size_{};
    feather::DataType input_dtype_{feather::DataType::UNKNOWN};
};
```

- [ ] **Step 4: Commit**

```bash
git add include/demo src/demo
git commit -m "feat: scaffold yolov5 demo helpers"
```

### Task 2: Add failing tests first

**Files:**
- Create: `test/yolov5_demo_test.cc`
- Test: `test/yolov5_demo_test.cc`

- [ ] **Step 1: Write a failing postprocess test**

```cpp
TEST(yolov5_demo_test, DecodeDetectionsAppliesConfidenceAndNms) {
    feather::Tensor output;
    output.Assign<float>({
        320.f, 320.f, 100.f, 120.f, 0.9f, 0.1f, 0.8f,
        322.f, 318.f, 100.f, 120.f, 0.88f, 0.1f, 0.79f,
        100.f, 100.f, 40.f,  30.f,  0.2f, 0.9f, 0.1f,
    }, {1, 3, 7});

    feather::demo::LetterboxInfo letterbox{640, 640, 0, 0, 1.0f};
    auto detections = feather::demo::DecodeYolov5Detections(output, letterbox, 640, 640, 0.25f, 0.45f);

    ASSERT_EQ(detections.size(), 1U);
    EXPECT_EQ(detections[0].class_id, 1);
    EXPECT_GT(detections[0].score, 0.7f);
}
```

- [ ] **Step 2: Write a failing preprocessing test**

```cpp
TEST(yolov5_demo_test, PreprocessImageBuildsNchwTensor) {
    feather::demo::ImageData image;
    image.width = 2;
    image.height = 2;
    image.channels = 3;
    image.pixels = {
        255, 0,   0,   0, 255, 0,
        0,   0, 255, 255, 255, 255,
    };

    std::shared_ptr<feather::Tensor> tensor;
    feather::demo::LetterboxInfo letterbox;
    ASSERT_EQ(feather::demo::PreprocessImageToTensor(image, 4, feather::DataType::FP32, &tensor, &letterbox), 0);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dims().data(), std::vector<int64_t>({1, 3, 4, 4}));
}
```

- [ ] **Step 3: Write a failing real-model integration test**

```cpp
TEST(yolov5_demo_test, LoadFthAndRunImageInference) {
    feather::demo::Yolov5Runner runner;
    ASSERT_EQ(runner.Load(output_path.string()), 0);
    std::vector<feather::demo::Detection> detections;
    ASSERT_EQ(runner.Run(image_path.string(), 0.25f, 0.45f, &detections), 0);
}
```

- [ ] **Step 4: Run the new test file and verify it fails**

```bash
cmake --build build -j4 --target unit_tests
./build/bin/unit_tests --gtest_filter='yolov5_demo_test.*'
```

Expected: compile errors or test failures because the demo helpers do not exist yet.

- [ ] **Step 5: Commit**

```bash
git add test/yolov5_demo_test.cc
git commit -m "test: add failing yolov5 demo tests"
```

### Task 3: Implement image decode and preprocessing

**Files:**
- Modify: `CMakeLists.txt`
- Create: `third_party/stb/stb_image.h`
- Modify: `include/demo/image_io.h`
- Modify: `src/demo/image_io.cc`

- [ ] **Step 1: Vendor `stb_image` and wire demo sources into the build**

```cmake
list(APPEND SRC
    "src/demo/image_io.cc"
    "src/demo/yolov5_postprocess.cc"
    "src/demo/yolov5_runner.cc")

add_executable(yolov5_demo demo/yolov5_infer_demo.cc)
target_link_libraries(yolov5_demo infer)
```

- [ ] **Step 2: Implement image loading and letterbox preprocessing**

```cpp
int32_t LoadImage(const std::string& path, ImageData* image);
int32_t PreprocessImageToTensor(const ImageData& image, int input_size, feather::DataType dtype,
                                std::shared_ptr<feather::Tensor>* tensor, LetterboxInfo* letterbox);
```

- [ ] **Step 3: Run preprocessing tests**

```bash
cmake --build build -j4 --target unit_tests
./build/bin/unit_tests --gtest_filter='yolov5_demo_test.PreprocessImageBuildsNchwTensor'
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt third_party/stb/stb_image.h include/demo/image_io.h src/demo/image_io.cc
git commit -m "feat: add image loading and preprocessing for demo"
```

### Task 4: Implement YOLOv5 decode and runner

**Files:**
- Modify: `include/demo/yolov5_postprocess.h`
- Modify: `include/demo/yolov5_runner.h`
- Modify: `src/demo/yolov5_postprocess.cc`
- Modify: `src/demo/yolov5_runner.cc`

- [ ] **Step 1: Implement decode, score filtering, and class-aware NMS**

```cpp
std::vector<Detection> DecodeYolov5Detections(const feather::Tensor& output, const LetterboxInfo& letterbox,
                                              int image_width, int image_height,
                                              float conf_thresh, float iou_thresh);
```

- [ ] **Step 2: Implement `.fth` loading and runtime execution**

```cpp
int32_t Yolov5Runner::Load(const std::string& model_path);
int32_t Yolov5Runner::Run(const std::string& image_path, float conf_thresh, float iou_thresh,
                          std::vector<Detection>* detections);
```

- [ ] **Step 3: Run targeted tests**

```bash
cmake --build build -j4 --target unit_tests
./build/bin/unit_tests --gtest_filter='yolov5_demo_test.DecodeDetectionsAppliesConfidenceAndNms:yolov5_demo_test.LoadFthAndRunImageInference'
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add include/demo/yolov5_postprocess.h include/demo/yolov5_runner.h src/demo/yolov5_postprocess.cc src/demo/yolov5_runner.cc
git commit -m "feat: add yolov5 demo runtime and postprocess"
```

### Task 5: Add the CLI demo entry and verify end-to-end

**Files:**
- Create: `demo/yolov5_infer_demo.cc`

- [ ] **Step 1: Implement CLI argument parsing and result printing**

```cpp
int main(int argc, char** argv) {
    feather::demo::Yolov5Runner runner;
    std::vector<feather::demo::Detection> detections;
    if (runner.Load(model_path) != 0 ||
        runner.Run(image_path, conf_thresh, iou_thresh, &detections) != 0) {
        return 1;
    }
    for (const auto& det : detections) {
        std::cout << det.class_id << " " << det.score << " "
                  << det.x1 << " " << det.y1 << " "
                  << det.x2 << " " << det.y2 << '\n';
    }
    return 0;
}
```

- [ ] **Step 2: Build the demo**

```bash
cmake --build build -j4 --target yolov5_demo
```

Expected: build succeeds with exit code `0`.

- [ ] **Step 3: Verify the real end-to-end demo**

```bash
./build/bin/yolov5_demo --model /tmp/yolov5n_demo.fth --image /path/to/test.jpg
```

Expected: program exits `0` and prints decoded detections or `detections: 0`.

- [ ] **Step 4: Commit**

```bash
git add demo/yolov5_infer_demo.cc
git commit -m "feat: add yolov5 fth image inference demo"
```

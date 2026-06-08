from conans import ConanFile, CMake, tools


class InferConan(ConanFile):
    name = "Infer"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "BUILD_UNITTEST": [True, False]}
    default_options = {"shared": False,
                       "fPIC": True,
                       "BUILD_UNITTEST": False}
    generators = "cmake"
    exports_sources = "CMakeLists.txt", "include*", "src*"
    install_folder = "./"

    def requirements(self):
        self.requires("benchmark/1.8.5")
        if self.options.BUILD_UNITTEST:
            self.requires("gtest/1.11.0")

    def configure(self):
        if self.settings.os == "Windows":
            del self.option.fPIC

    def configure_cmake(self):
        cmake = CMake(self)
        cmake.configure()
        return cmake

    def build(self):
        cmake = self.configure_cmake()
        cmake.build()

    def package(self):
        cmake = self.configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)

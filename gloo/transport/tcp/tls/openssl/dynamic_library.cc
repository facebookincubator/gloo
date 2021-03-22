#include "dynamic_library.h"

#include <stdexcept>
#include <dlfcn.h>

DynamicLibrary::DynamicLibrary(const char *name, const char *alt_name) : lib_name(name) {
  handle = dlopen(name, RTLD_LOCAL | RTLD_NOW);
  if (!handle) {
    handle = dlopen(alt_name, RTLD_LOCAL | RTLD_NOW);
  }
}

bool DynamicLibrary::loaded() const {
  return handle != nullptr;
}

void *DynamicLibrary::sym(const char *name) {
  if (loaded_set.count(name)) {
    throw std::runtime_error(std::string(name) + " already loaded");
  }
  loaded_set.insert(name);
  void* res = dlsym(handle, name);
  if (res == nullptr) {
    throw std::runtime_error("Can't find " + std::string(name) + " in " + std::string(lib_name));
  }
  return res;
}

DynamicLibrary::~DynamicLibrary() {
  if (!handle)
    return;
  dlclose(handle);
}

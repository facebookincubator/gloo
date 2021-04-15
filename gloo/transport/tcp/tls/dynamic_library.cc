#include "dynamic_library.h"

#include <dlfcn.h>
#include <stdexcept>

DynamicLibrary::DynamicLibrary(const char *name, const char *alt_name)
    : lib_name(name) {
  handle = dlopen(name, RTLD_LOCAL | RTLD_NOW);
  if (!handle) {
    if (alt_name == nullptr) {
      throw std::runtime_error(dlerror());
    } else {
      handle = dlopen(alt_name, RTLD_LOCAL | RTLD_NOW);
      if (!handle) {
        throw std::runtime_error(dlerror());
      }
    }
  }
}

void *DynamicLibrary::sym(const char *name) {
  void *res = dlsym(handle, name);
  if (res == nullptr) {
    throw std::runtime_error("Can't find " + std::string(name) + " in " +
                             lib_name + ":" + dlerror());
  }
  return res;
}

DynamicLibrary::~DynamicLibrary() {
  if (!handle)
    return;
  dlclose(handle);
}

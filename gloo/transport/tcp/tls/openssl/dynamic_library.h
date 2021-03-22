#include <unordered_set>

struct DynamicLibrary {
  DynamicLibrary(const DynamicLibrary&) = delete;

  void operator=(const DynamicLibrary&) = delete;

  DynamicLibrary(const char* name, const char* alt_name);

  bool loaded() const;

  void* sym(const char* name);

  ~DynamicLibrary();

private:
  const char* lib_name;
  void* handle = nullptr;
  std::unordered_set<std::string> loaded_set;
};

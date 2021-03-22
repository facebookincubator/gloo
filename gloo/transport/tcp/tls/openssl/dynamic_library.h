struct DynamicLibrary {
  DynamicLibrary(const DynamicLibrary&) = delete;

  void operator=(const DynamicLibrary&) = delete;

  DynamicLibrary(const char* name, const char* alt_name);

  void* sym(const char* name);

  ~DynamicLibrary();

private:
  const char* lib_name;
  void* handle = nullptr;
  const char* error = nullptr;
};

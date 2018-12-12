/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "gloo/context.h"
#include "gloo/transport/device.h"

#include <mpi.h>

namespace gloo {
namespace mpi {

class MPIScope {
 public:
  MPIScope();
  ~MPIScope();
};

class Context : public ::gloo::Context {
 public:
  // This function acquires and holds on to a global MPI scope object.
  // The MPI scope object calls MPI_Init upon construction and
  // MPI_Finalize upon destruction. Use this function to create a
  // context if you want this to be managed by Gloo.
  static std::shared_ptr<Context> createManaged();

  // This constructor clone the specified MPI common world. Use it if
  // you are calling MPI_Init and MPI_Finalize yourself.
  explicit Context(const MPI_Comm& comm);

  virtual ~Context();

  void connectFullMesh(std::shared_ptr<transport::Device>& dev);

 protected:
  // If Gloo is responsible for calling MPI_Init and MPI_Finalize,
  // this context refers to a singleton initializer. As soon as the
  // last MPI context is destructed, this initializer is destructed,
  // and MPI_Finalize will be called.
  std::shared_ptr<MPIScope> mpiScope_;

  MPI_Comm comm_;
};

} // namespace mpi
} // namespace gloo

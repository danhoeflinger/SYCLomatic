//==---------- ur.hpp - Unified Runtime integration helpers ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
///
/// C++ utilities for Unified Runtime integration.
///
/// \ingroup sycl_ur

#pragma once

#include <sycl/backend_types.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/os_util.hpp>
#
#include <ur_api.h>

#include <memory>
#include <type_traits>
#include <vector>

/// Extension to denote native support of assert feature by an arbitrary device
/// urDeviceGetInfo call should return this extension when the device supports
/// native asserts if supported extensions' names are requested
#define UR_DEVICE_INFO_EXTENSION_DEVICELIB_ASSERT "cl_intel_devicelib_assert"

typedef void (*pi_context_extended_deleter)(void *user_data);

struct _sycl_device_binary_property_struct;
using sycl_device_binary_property = _sycl_device_binary_property_struct*;

#ifdef XPTI_ENABLE_INSTRUMENTATION
// Forward declarations
namespace xpti {
struct trace_event_data_t;
}
#endif

namespace sycl {
inline namespace _V1 {

class context;

namespace detail {

namespace pi {
// This function is deprecated and it should be removed in the next release
// cycle (along with the definition for pi_context_extended_deleter).
__SYCL_EXPORT void contextSetExtendedDeleter(const sycl::context &constext,
                                             pi_context_extended_deleter func,
                                             void *user_data);
}

class plugin;
using PluginPtr = std::shared_ptr<plugin>;

// TODO: To be removed as this was only introduced for esimd which was removed.
template <sycl::backend BE>
__SYCL_EXPORT void *getPluginOpaqueData(void *opaquedata_arg);

namespace ur {
// Function to load a shared library
// Implementation is OS dependent
void *loadOsLibrary(const std::string &Library);

// Function to unload a shared library
// Implementation is OS dependent (see posix-ur.cpp and windows-ur.cpp)
int unloadOsLibrary(void *Library);

// Function to get Address of a symbol defined in the shared
// library, implementation is OS dependent.
void *getOsLibraryFuncAddress(void *Library, const std::string &FunctionName);

// Performs UR one-time initialization.
std::vector<PluginPtr> &
initializeUr(ur_loader_config_handle_t LoaderConfig = nullptr);

// Get the plugin serving given backend.
template <backend BE> const PluginPtr &getPlugin();

// Return true if we want to trace UR related activities.
bool trace();

// Want all the needed casts be explicit, do not define conversion operators.
template <class To, class From> To cast(From value);

// Want all the needed casts be explicit, do not define conversion
// operators.
template <class To, class From> inline To cast(From value) {
  // TODO: see if more sanity checks are possible.
  static_assert(sizeof(From) == sizeof(To), "assert: cast failed size check");
  return reinterpret_cast<To>(value);
}

// Helper traits for identifying std::vector with arbitrary element type.
template <typename T> struct IsStdVector : std::false_type {};
template <typename T> struct IsStdVector<std::vector<T>> : std::true_type {};

// Overload for vectors that applies the cast to all elements. This
// creates a new vector.
template <class To, class FromE> To cast(std::vector<FromE> Values) {
  static_assert(IsStdVector<To>::value, "Return type must be a vector.");
  To ResultVec;
  ResultVec.reserve(Values.size());
  for (FromE &Val : Values) {
    ResultVec.push_back(cast<typename To::value_type>(Val));
  }
  return ResultVec;
}

ur_program_metadata_t mapDeviceBinaryPropertyToProgramMetadata(
    const sycl_device_binary_property &DeviceBinaryProperty);

} // namespace ur
} // namespace detail
} // namespace _V1
} // namespace sycl

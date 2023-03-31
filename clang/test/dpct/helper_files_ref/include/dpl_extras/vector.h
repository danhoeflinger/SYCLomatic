//==---- vector.h ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_VECTOR_H__
#define __DPCT_VECTOR_H__

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include <sycl/sycl.hpp>

#include "memory.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include "../device.hpp"

namespace dpct {

namespace internal {

// usm_device_allocator is provided here specifically for dpct::device_vector.
//  Warning: It may be dangerous to use usm_device_allocator in other settings,
//  because containers may use the supplied allocator to allocate side
//  information which needs to be available on the host.  Data allocated with
//  this allocator is by definition not available on the host, and would result
//  in an error if accessed from the host without proper handling.
template <typename T, size_t Alignment = 0> class usm_device_allocator {
public:
  using value_type = T;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

public:
  template <typename U> struct rebind {
    typedef usm_device_allocator<U, Alignment> other;
  };

  usm_device_allocator() = delete;
  usm_device_allocator(const sycl::context &Ctxt, const sycl::device &Dev,
                       const sycl::property_list &PropList = {})
      : MContext(Ctxt), MDevice(Dev), MPropList(PropList) {}
  usm_device_allocator(const sycl::queue &Q,
                       const sycl::property_list &PropList = {})
      : MContext(Q.get_context()), MDevice(Q.get_device()),
        MPropList(PropList) {}
  usm_device_allocator(const usm_device_allocator &) = default;
  usm_device_allocator(usm_device_allocator &&) noexcept = default;
  usm_device_allocator &operator=(const usm_device_allocator &Other) {
    MContext = Other.MContext;
    MDevice = Other.MDevice;
    MPropList = Other.MPropList;
    return *this;
  }
  usm_device_allocator &operator=(usm_device_allocator &&Other) {
    MContext = std::move(Other.MContext);
    MDevice = std::move(Other.MDevice);
    MPropList = std::move(Other.MPropList);
    return *this;
  }

  template <class U>
  usm_device_allocator(const usm_device_allocator<U, Alignment> &Other) noexcept
      : MContext(Other.MContext), MDevice(Other.MDevice),
        MPropList(Other.MPropList) {}

  /// Allocates memory.
  ///
  /// \param NumberOfElements is a count of elements to allocate memory for.
  T *allocate(size_t NumberOfElements,
              const sycl::detail::code_location CodeLoc =
                  sycl::detail::code_location::current()) {

    auto Result = reinterpret_cast<T *>(aligned_alloc(
        getAlignment(), NumberOfElements * sizeof(value_type), MDevice,
        MContext, sycl::usm::alloc::device, MPropList, CodeLoc));
    if (!Result) {
      throw sycl::exception(sycl::errc::memory_allocation);
    }
    return Result;
  }

  /// Deallocates memory.
  ///
  /// \param Ptr is a pointer to memory being deallocated.
  /// \param Size is a number of elements previously passed to allocate.
  void deallocate(T *Ptr, size_t,
                  const sycl::detail::code_location CodeLoc =
                      sycl::detail::code_location::current()) {
    if (Ptr) {
      free(Ptr, MContext, CodeLoc);
    }
  }

  template <class U, size_t AlignmentU>
  friend bool operator==(const usm_device_allocator<T, Alignment> &One,
                         const usm_device_allocator<U, AlignmentU> &Two) {
    return ((One.MContext == Two.MContext) && (One.MDevice == Two.MDevice));
  }

  template <class U, size_t AlignmentU>
  friend bool operator!=(const usm_device_allocator<T, Alignment> &One,
                         const usm_device_allocator<U, AlignmentU> &Two) {
    return !((One.MContext == Two.MContext) && (One.MDevice == Two.MDevice));
  }

  template <typename Property> bool has_property() const noexcept {
    return MPropList.has_property<Property>();
  }

  template <typename Property> Property get_property() const {
    return MPropList.get_property<Property>();
  }

private:
  constexpr size_t getAlignment() const {
    return sycl::max(alignof(T), Alignment);
  }

  template <class U, size_t AlignmentU> friend class usm_device_allocator;

  sycl::context MContext;
  sycl::device MDevice;
  sycl::property_list MPropList;
};

template <class, class _Alloc, class... _Args>
struct __has_construct_impl : ::std::false_type {};

template <class _Alloc, class... _Args>
struct __has_construct_impl<decltype((void)std::declval<_Alloc>().construct(
                                std::declval<_Args>()...)),
                            _Alloc, _Args...> : ::std::true_type {};

// check if the provided allocator has a construct() member function
template <class _Alloc, class... _Args>
struct __has_construct : __has_construct_impl<void, _Alloc, _Args...> {};

// check if the provided allocator has a destroy() member function
template <class _Alloc, class _Pointer, class = void>
struct __has_destroy : ::std::false_type {};

template <class _Alloc, class _Pointer>
struct __has_destroy<_Alloc, _Pointer,
                     decltype((void)std::declval<_Alloc>().destroy(
                         std::declval<_Pointer>()))> : ::std::true_type {};

// device_allocator_traits is a device-friendly subset of the functionality of
//  std::allocator_traits which uses static construct and destroy functions
//  and is usable inside of sycl kernels without passing the allocator to the
//  kernel.
template <typename _Allocator> struct device_allocator_traits {

  // apply default constructor if no override is provided
  template <typename DataT>
  static
      typename ::std::enable_if_t<!__has_construct<_Allocator, DataT *>::value,
                                  void>
      construct(DataT *p) {
    ::new ((void *)p) DataT();
  }

  // use provided default construct call if it exists
  template <typename DataT>
  static
      typename ::std::enable_if_t<__has_construct<_Allocator, DataT *>::value,
                                  void>
      construct(DataT *p) {
    _Allocator::construct(p);
  }

  // apply constructor if no override is provided
  template <typename DataT, typename T_in>
  static typename ::std::enable_if_t<
      !__has_construct<_Allocator, DataT *, T_in>::value, void>
  construct(DataT *p, T_in arg) {
    ::new ((void *)p) DataT(arg);
  }

  // use provided construct call if it exists
  template <typename DataT, typename T_in>
  static typename ::std::enable_if_t<
      __has_construct<_Allocator, DataT *, T_in>::value, void>
  construct(DataT *p, T_in arg) {
    _Allocator::construct(p, arg);
  }

  // apply default destructor if no destroy override is provided
  template <typename DataT>
  static typename ::std::enable_if_t<!__has_destroy<_Allocator, DataT *>::value,
                                     void>
  destroy(DataT *p) {
    p->~DataT();
  }

  // use provided destroy call if it exists
  template <typename DataT>
  static typename ::std::enable_if_t<__has_destroy<_Allocator, DataT *>::value,
                                     void>
  destroy(DataT *p) {
    _Allocator::destroy(p);
  }
};

template <typename Iter, typename Void = void> // for non-iterators
struct is_iterator : std::false_type {};

template <typename Iter> // For iterators
struct is_iterator<
    Iter,
    typename std::enable_if<
        !std::is_void<typename Iter::iterator_category>::value, void>::type>
    : std::true_type {};

template <typename T> // For pointers
struct is_iterator<T *> : std::true_type {};
} // end namespace internal

#ifndef DPCT_USM_LEVEL_NONE

template <typename T,
          typename Allocator = dpct::internal::usm_device_allocator<T>>
class device_vector {
public:
  using iterator = device_iterator<T>;
  using const_iterator = const iterator;
  using reference = device_reference<T>;
  using const_reference = const reference;
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using difference_type =
      typename ::std::iterator_traits<iterator>::difference_type;
  using size_type = ::std::size_t;
  using allocator_type = Allocator;

private:
  Allocator _alloc;
  size_type _size;
  size_type _capacity;
  pointer _storage;

  size_type _min_capacity() const { return size_type(1); }

  void _set_capacity_and_alloc() {
    _capacity = ::std::max(_size * 2, _min_capacity());
    _storage = ::std::allocator_traits<Allocator>::allocate(_alloc, _capacity);
  }

  void _construct(size_type n, size_type start_idx = 0) {
    if (n > 0) {
      pointer p = _storage;
      get_default_queue()
          .submit([&](sycl::handler &cgh) {
            cgh.parallel_for(n, [=](sycl::id<1> i) {
              ::dpct::internal::device_allocator_traits<Allocator>::construct(
                  p + start_idx + i);
            });
          })
          .wait();
    }
  }

  void _construct(size_type n, const T &value, size_type start_idx = 0) {
    if (n > 0) {
      pointer p = _storage;
      get_default_queue()
          .submit([&](sycl::handler &cgh) {
            cgh.parallel_for(n, [=](sycl::id<1> i) {
              ::dpct::internal::device_allocator_traits<Allocator>::construct(
                  p + start_idx + i, value);
            });
          })
          .wait();
    }
  }

  template <typename Iter>
  void _construct(Iter first, Iter last, size_type start_idx = 0) {
    int num_eles = ::std::distance(first, last);
    if (num_eles > 0) {
      // this should properly handle host or device input data
      auto read_input =
          oneapi::dpl::__ranges::__get_sycl_range<sycl::access_mode::read,
                                                  Iter>();
      auto input_rng = read_input(first, last).all_view();
      pointer p = _storage;
      get_default_queue()
          .submit([&](sycl::handler &cgh) {
            oneapi::dpl::__ranges::__require_access(cgh, input_rng);
            cgh.parallel_for(num_eles, [=](sycl::id<1> i) {
              ::dpct::internal::device_allocator_traits<Allocator>::construct(
                  p + start_idx + i, input_rng[i]);
            });
          })
          .wait();
    }
  }

  void _destroy(size_type n, size_type start_idx = 0) {
    // only call destroy kernel *only* if custom destroy function is provided to
    //  prevent extra unnecessary kernel call
    if constexpr (::dpct::internal::__has_destroy<Allocator, pointer>::value) {
      if (n > 0) {
        pointer p = _storage;
        get_default_queue()
            .submit([&](sycl::handler &cgh) {
              cgh.parallel_for(n, [=](sycl::id<1> i) {
                ::dpct::internal::device_allocator_traits<Allocator>::destroy(
                    p + start_idx + i);
              });
            })
            .wait();
      }
    }
  }

public:
  template <typename OtherA> operator ::std::vector<T, OtherA>() const {
    auto __tmp = ::std::vector<T, OtherA>(this->size());
    ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                this->begin(), this->end(), __tmp.begin());
    return __tmp;
  }
  device_vector()
      : _alloc(get_default_queue()), _size(0), _capacity(_min_capacity()) {
    _set_capacity_and_alloc();
  }
  ~device_vector() /*= default*/ {
    _destroy(size());
    ::std::allocator_traits<Allocator>::deallocate(_alloc, _storage, _capacity);
  }
  explicit device_vector(size_type n) : _alloc(get_default_queue()), _size(n) {
    _set_capacity_and_alloc();
    _construct(n);
  }
  explicit device_vector(size_type n, const T &value)
      : _alloc(get_default_queue()), _size(n) {
    _set_capacity_and_alloc();
    _construct(n, value);
  }

  device_vector(device_vector &&other)
      : _alloc(std::move(other._alloc)), _size(other.size()),
        _capacity(other.capacity()), _storage(other._storage) {
    other._size = 0;
    other._capacity = 0;
    other._storage = nullptr;
  }

  template <typename InputIterator>
  device_vector(InputIterator first, 
  typename ::std::enable_if_t<dpct::internal::is_iterator<InputIterator>::value, InputIterator> last)
      : _alloc(get_default_queue()) {
    _size = ::std::distance(first, last);
    _set_capacity_and_alloc();
    _construct(first, last);
  }

  device_vector(const device_vector &other)
      : _alloc(std::allocator_traits<Allocator>::select_on_container_copy_construction(other._alloc)) {
    _size = other.size();
    _capacity = other.capacity();
    _storage = ::std::allocator_traits<Allocator>::allocate(_alloc, _capacity);
    _construct(other.begin(), other.end());
  }

  template <typename OtherAllocator>
  device_vector(const device_vector<T, OtherAllocator> &other)
      : _alloc(get_default_queue()) {
    _size = other.size();
    _capacity = other.capacity();
    _storage = ::std::allocator_traits<Allocator>::allocate(_alloc, _capacity);
    _construct(other.begin(), other.end());
  }

  template <typename OtherAllocator>
  device_vector(::std::vector<T, OtherAllocator> &v)
      : device_vector(v.begin(), v.end()) {}
  template <typename OtherAllocator>
  device_vector &operator=(const ::std::vector<T, OtherAllocator> &v) {
    resize(v.size());
    _construct(v.begin(), v.end());
    return *this;
  }
  device_vector &operator=(const device_vector &other) {
    // Copy assignment operator:
    if constexpr(::std::allocator_traits<Allocator>::propagate_on_container_copy_assignment::value)
    {
      _alloc = other._alloc;
    }
    resize(other.size());
    _construct(other.begin(), other.end());
    return *this;
  }
  device_vector &operator=(device_vector &&other) {
    // Move assignment operator:
    if constexpr(::std::allocator_traits<Allocator>::propagate_on_container_move_assignment::value)
    {
      _alloc = std::move(other._alloc);
    }
    _size = std::move(other._size);
    _capacity = std::move(other._capacity);
    _storage = std::move(other._storage);
    return *this;
  }
  size_type size() const { return _size; }
  iterator begin() noexcept { return device_iterator<T>(_storage, 0); }
  iterator end() { return device_iterator<T>(_storage, size()); }
  const_iterator begin() const noexcept {
    return device_iterator<T>(_storage, 0);
  }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator end() const { return device_iterator<T>(_storage, size()); }
  const_iterator cend() const { return end(); }
  T *real_begin() { return _storage; }
  const T *real_begin() const { return _storage; }
  void swap(device_vector &v) {
    ::std::swap(_size, v._size);
    ::std::swap(_capacity, v._capacity);
    ::std::swap(_storage, v._storage);
    if constexpr (::std::allocator_traits<Allocator>::propagate_on_container_swap::value)
    {
      ::std::swap(_alloc, v._alloc);
    }
  }
  reference operator[](size_type n) { return _storage[n]; }
  const_reference operator[](size_type n) const { return _storage[n]; }
  void reserve(size_type n) {
    if (n > capacity()) {
      // allocate buffer for new size
      auto tmp = ::std::allocator_traits<Allocator>::allocate(_alloc, 2 * n);
      // copy content (old buffer to new buffer)
      ::std::copy(
          oneapi::dpl::execution::make_device_policy(get_default_queue()),
          begin(), end(), tmp);
      // deallocate old memory
      ::std::allocator_traits<Allocator>::deallocate(_alloc, _storage,
                                                     _capacity);
      _storage = tmp;
      _capacity = 2 * n;
    }
  }
  void resize(size_type new_size, const T &x = T()) {
    reserve(new_size);
    if (new_size > size()) {
      _construct(new_size - size(), x, size());
    }
    _size = new_size;
  }
  size_type max_size(void) const {
    return ::std::numeric_limits<size_type>::max() / sizeof(T);
  }
  size_type capacity() const { return _capacity; }
  const_reference front() const { return *begin(); }
  reference front() { return *begin(); }
  const_reference back(void) const { return *(end() - 1); }
  reference back(void) { return *(end() - 1); }
  pointer data(void) { return _storage; }
  const_pointer data(void) const { return _storage; }
  void shrink_to_fit(void) {
    if (_size != capacity()) {
      size_type tmp_capacity = ::std::max(_size, _min_capacity());
      auto tmp =
          ::std::allocator_traits<Allocator>::allocate(_alloc, tmp_capacity);
      if (_size > 0) {
        ::std::copy(
            oneapi::dpl::execution::make_device_policy(get_default_queue()),
            begin(), end(), tmp);
      }
      ::std::allocator_traits<Allocator>::deallocate(_alloc, _storage,
                                                     _capacity);
      _storage = tmp;
      _capacity = tmp_capacity;
    }
  }
  void assign(size_type n, const T &x) {
    resize(n);
    if (_size > 0) {
      ::std::fill(
          oneapi::dpl::execution::make_device_policy(get_default_queue()),
          begin(), begin() + n, x);
    }
  }
  template <typename InputIterator>
  void
  assign(InputIterator first,
         typename ::std::enable_if<internal::is_iterator<InputIterator>::value,
                                   InputIterator>::type last) {
    auto n = ::std::distance(first, last);
    resize(n);
    if (_size > 0) {
      ::std::copy(
          oneapi::dpl::execution::make_device_policy(get_default_queue()),
          first, last, begin());
    }
  }
  void clear(void) { _size = 0; }
  bool empty(void) const { return (size() == 0); }
  void push_back(const T &x) { insert(end(), size_type(1), x); }
  void pop_back(void) {
    if (_size > 0)
      --_size;
  }
  iterator erase(iterator first, iterator last) {
    auto n = ::std::distance(first, last);
    if (last == end()) {
      _size = _size - n;
      return end();
    }
    auto m = ::std::distance(last, end());
    if (m <= 0) {
      return end();
    }
    auto tmp = ::std::allocator_traits<Allocator>::allocate(_alloc, m);
    // copy remainder to temporary buffer.
    ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                last, end(), tmp);

    auto position = ::std::distance(begin(), first);
    _destroy(n, position);

    // override (erase) subsequence in storage.
    ::std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
                tmp, tmp + m, first);
    ::std::allocator_traits<Allocator>::deallocate(_alloc, tmp, m);
    _size -= n;
    return begin() + first.get_idx() + n;
  }
  iterator erase(iterator pos) { return erase(pos, pos + 1); }
  iterator insert(iterator position, const T &x) {
    auto n = ::std::distance(begin(), position);
    insert(position, size_type(1), x);
    return begin() + n;
  }
  void insert(iterator position, size_type n, const T &x) {
    if (position == end()) {
      resize(size() + n);
      _construct(n, x, size() - n);
    } else {
      auto i_n = ::std::distance(begin(), position);
      // allocate temporary storage
      auto m = ::std::distance(position, end());
      // will throw if position is not inside active vector
      auto tmp = ::std::allocator_traits<Allocator>::allocate(_alloc, m);
      // copy remainder
      ::std::copy(
          oneapi::dpl::execution::make_device_policy(get_default_queue()),
          position, end(), tmp);

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      _construct(n, x, position.get_idx());

      ::std::copy(
          oneapi::dpl::execution::make_device_policy(get_default_queue()), tmp,
          tmp + m, position + n);
      ::std::allocator_traits<Allocator>::deallocate(_alloc, tmp, m);
    }
  }
  template <typename InputIterator>
  void
  insert(iterator position, InputIterator first,
         typename ::std::enable_if<internal::is_iterator<InputIterator>::value,
                                   InputIterator>::type last) {
    auto n = ::std::distance(first, last);
    if (position == end()) {
      resize(size() + n);
      _construct(first, last, size() - n);
      ::std::copy(
          oneapi::dpl::execution::make_device_policy(get_default_queue()),
          first, last, end());
    } else {
      auto m = ::std::distance(position, end());
      // will throw if position is not inside active vector
      auto tmp = ::std::allocator_traits<Allocator>::allocate(_alloc, m);

      ::std::copy(
          oneapi::dpl::execution::make_device_policy(get_default_queue()),
          position, end(), tmp);

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      _construct(first, last, position.get_idx());

      ::std::copy(
          oneapi::dpl::execution::make_device_policy(get_default_queue()), tmp,
          tmp + m, position + n);
      ::std::allocator_traits<Allocator>::deallocate(_alloc, tmp, m);
    }
  }
  Allocator get_allocator() const { return _alloc; }
};

#else

template <typename T, typename Allocator = detail::__buffer_allocator<T>>
class device_vector {
  static_assert(
      std::is_same<Allocator, detail::__buffer_allocator<T>>::value,
      "device_vector doesn't support custom allocator when USM is not used.");

public:
  using iterator = device_iterator<T>;
  using const_iterator = const iterator;
  using reference = device_reference<T>;
  using const_reference = const reference;
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using difference_type =
      typename std::iterator_traits<iterator>::difference_type;
  using size_type = std::size_t;

private:
  using Buffer = sycl::buffer<T, 1>;
  using Range = sycl::range<1>;
  // Using mem_mgr to handle memory allocation
  void *_storage;
  size_type _size;

  size_type _min_capacity() const { return size_type(1); }

  void *alloc_store(size_type num_bytes) {
    return detail::mem_mgr::instance().mem_alloc(num_bytes);
  }

public:
  template <typename OtherA> operator std::vector<T, OtherA>() const {
    auto __tmp = std::vector<T, OtherA>(this->size());
    std::copy(oneapi::dpl::execution::dpcpp_default, this->begin(), this->end(),
              __tmp.begin());
    return __tmp;
  }
  device_vector()
      : _storage(alloc_store(_min_capacity() * sizeof(T))), _size(0) {}
  ~device_vector() = default;
  explicit device_vector(size_type n) : device_vector(n, T()) {}
  explicit device_vector(size_type n, const T &value)
      : _storage(alloc_store(std::max(n, _min_capacity()) * sizeof(T))),
        _size(n) {
    auto buf = get_buffer();
    std::fill(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(buf),
              oneapi::dpl::begin(buf) + n, T(value));
  }
  device_vector(const device_vector &other)
      : _storage(other._storage), _size(other.size()) {}
  device_vector(device_vector &&other)
      : _storage(std::move(other._storage)), _size(other.size()) {}

  template <typename InputIterator>
  device_vector(InputIterator first,
                typename std::enable_if<
                    internal::is_iterator<InputIterator>::value &&
                        !std::is_pointer<InputIterator>::value &&
                        std::is_same<typename std::iterator_traits<
                                         InputIterator>::iterator_category,
                                     std::random_access_iterator_tag>::value,
                    InputIterator>::type last)
      : _storage(alloc_store(std::distance(first, last) * sizeof(T))),
        _size(std::distance(first, last)) {
    auto buf = get_buffer();
    auto dst = oneapi::dpl::begin(buf);
    std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
              first, last, dst);
  }

  template <typename InputIterator>
  device_vector(InputIterator first,
                typename std::enable_if<std::is_pointer<InputIterator>::value,
                                        InputIterator>::type last)
      : _storage(alloc_store(std::distance(first, last) * sizeof(T))),
        _size(std::distance(first, last)) {
    auto buf = get_buffer();
    Buffer tmp_buf(first, last);
    auto start = oneapi::dpl::begin(tmp_buf);
    auto end = oneapi::dpl::end(tmp_buf);
    auto dst = oneapi::dpl::begin(buf);
    std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
              start, end, dst);
  }

  template <typename InputIterator>
  device_vector(InputIterator first,
                typename std::enable_if<
                    internal::is_iterator<InputIterator>::value &&
                        !std::is_same<typename std::iterator_traits<
                                          InputIterator>::iterator_category,
                                      std::random_access_iterator_tag>::value,
                    InputIterator>::type last)
      : _storage(alloc_store(std::distance(first, last) * sizeof(T))),
        _size(std::distance(first, last)) {
    auto buf = get_buffer();
    std::vector<T> tmp(first, last);
    Buffer tmp_buf(tmp);
    auto start = oneapi::dpl::begin(tmp_buf);
    auto end = oneapi::dpl::end(tmp_buf);
    auto dst = oneapi::dpl::begin(buf);
    std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
              start, end, dst);
  }

  template <typename OtherAllocator>
  device_vector(const device_vector<T, OtherAllocator> &v)
      : _storage(alloc_store(v.size() * sizeof(T))), _size(v.size()) {
    auto buf = get_buffer();
    auto dst = oneapi::dpl::begin(buf);
    std::copy(oneapi::dpl::execution::make_device_policy(get_default_queue()),
              v.real_begin(), v.real_begin() + v.size(), dst);
  }

  template <typename OtherAllocator>
  device_vector(std::vector<T, OtherAllocator> &v)
      : _storage(alloc_store(v.size() * sizeof(T))), _size(v.size()) {
    std::copy(oneapi::dpl::execution::dpcpp_default, v.begin(), v.end(),
              oneapi::dpl::begin(get_buffer()));
  }

  device_vector &operator=(const device_vector &other) {
    // Copy assignment operator:
    _size = other.size();
    void *tmp = alloc_store(_size * sizeof(T));
    auto tmp_buf =
        detail::mem_mgr::instance()
            .translate_ptr(tmp)
            .buffer.template reinterpret<T, 1>(sycl::range<1>(_size));
    std::copy(oneapi::dpl::execution::dpcpp_default,
              oneapi::dpl::begin(other.get_buffer()),
              oneapi::dpl::end(other.get_buffer()),
              oneapi::dpl::begin(tmp_buf));
    detail::mem_mgr::instance().mem_free(_storage);
    _storage = tmp;
    return *this;
  }
  device_vector &operator=(device_vector &&other) {
    // Move assignment operator:
    _size = other.size();
    this->_storage = std::move(other._storage);
    return *this;
  }
  template <typename OtherAllocator>
  device_vector &operator=(const std::vector<T, OtherAllocator> &v) {
    Buffer data(v.begin(), v.end());
    _size = v.size();
    void *tmp = alloc_store(_size * sizeof(T));
    auto tmp_buf =
        detail::mem_mgr::instance()
            .translate_ptr(tmp)
            .buffer.template reinterpret<T, 1>(sycl::range<1>(_size));
    std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(data),
              oneapi::dpl::end(data), oneapi::dpl::begin(tmp_buf));
    detail::mem_mgr::instance().mem_free(_storage);
    _storage = tmp;

    return *this;
  }
  Buffer get_buffer() const {
    return detail::mem_mgr::instance()
        .translate_ptr(_storage)
        .buffer.template reinterpret<T, 1>(sycl::range<1>(capacity()));
  }
  size_type size() const { return _size; }
  iterator begin() noexcept { return device_iterator<T>(get_buffer(), 0); }
  iterator end() { return device_iterator<T>(get_buffer(), _size); }
  const_iterator begin() const noexcept {
    return device_iterator<T>(get_buffer(), 0);
  }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator end() const { return device_iterator<T>(get_buffer(), _size); }
  const_iterator cend() const { return end(); }
  T *real_begin() {
    return (detail::mem_mgr::instance()
                .translate_ptr(_storage)
                .buffer.template get_access<sycl::access_mode::read_write>())
        .get_pointer();
  }
  const T *real_begin() const {
    return const_cast<device_vector *>(this)
        ->detail::mem_mgr::instance()
        .translate_ptr(_storage)
        .buffer.template get_access<sycl::access_mode::read_write>()
        .get_pointer();
  }
  void swap(device_vector &v) {
    void *temp = v._storage;
    v._storage = this->_storage;
    this->_storage = temp;
    std::swap(_size, v._size);
  }
  reference operator[](size_type n) { return *(begin() + n); }
  const_reference operator[](size_type n) const { return *(begin() + n); }
  void reserve(size_type n) {
    if (n > capacity()) {
      // create new buffer (allocate for new size)
      void *a = alloc_store(n * sizeof(T));

      // copy content (old buffer to new buffer)
      if (_storage != nullptr) {
        auto tmp = detail::mem_mgr::instance()
                       .translate_ptr(a)
                       .buffer.template reinterpret<T, 1>(sycl::range<1>(n));
        auto src_buf = get_buffer();
        std::copy(oneapi::dpl::execution::dpcpp_default,
                  oneapi::dpl::begin(src_buf), oneapi::dpl::end(src_buf),
                  oneapi::dpl::begin(tmp));

        // deallocate old memory
        detail::mem_mgr::instance().mem_free(_storage);
      }
      _storage = a;
    }
  }
  void resize(size_type new_size, const T &x = T()) {
    reserve(new_size);
    if (_size < new_size) {
      auto src_buf = get_buffer();
      std::fill(oneapi::dpl::execution::dpcpp_default,
                oneapi::dpl::begin(src_buf) + _size,
                oneapi::dpl::begin(src_buf) + new_size, x);
    }
    _size = new_size;
  }
  size_type max_size(void) const {
    return std::numeric_limits<size_type>::max() / sizeof(T);
  }
  size_type capacity() const {
    return _storage != nullptr ? detail::mem_mgr::instance()
                                         .translate_ptr(_storage)
                                         .buffer.size() /
                                     sizeof(T)
                               : 0;
  }
  const_reference front() const { return *begin(); }
  reference front() { return *begin(); }
  const_reference back(void) const { return *(end() - 1); }
  reference back(void) { return *(end() - 1); }
  pointer data(void) { return reinterpret_cast<pointer>(_storage); }
  const_pointer data(void) const {
    return reinterpret_cast<const_pointer>(_storage);
  }
  void shrink_to_fit(void) {
    if (_size != capacity()) {
      void *a = alloc_store(_size * sizeof(T));
      auto tmp = detail::mem_mgr::instance()
                     .translate_ptr(a)
                     .buffer.template reinterpret<T, 1>(sycl::range<1>(_size));
      std::copy(oneapi::dpl::execution::dpcpp_default,
                oneapi::dpl::begin(get_buffer()),
                oneapi::dpl::begin(get_buffer()) + _size,
                oneapi::dpl::begin(tmp));
      detail::mem_mgr::instance().mem_free(_storage);
      _storage = a;
    }
  }
  void assign(size_type n, const T &x) {
    resize(n);
    std::fill(oneapi::dpl::execution::dpcpp_default, begin(), begin() + n, x);
  }
  template <typename InputIterator>
  void
  assign(InputIterator first,
         typename std::enable_if<internal::is_iterator<InputIterator>::value,
                                 InputIterator>::type last) {
    auto n = std::distance(first, last);
    resize(n);
    if (internal::is_iterator<InputIterator>::value &&
        !std::is_pointer<InputIterator>::value)
      std::copy(oneapi::dpl::execution::dpcpp_default, first, last, begin());
    else {
      Buffer tmp(first, last);
      std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(tmp),
                oneapi::dpl::end(tmp), begin());
    }
  }
  void clear(void) {
    _size = 0;
    detail::mem_mgr::instance().mem_free(_storage);
    _storage = nullptr;
  }
  bool empty(void) const { return (size() == 0); }
  void push_back(const T &x) { insert(end(), size_type(1), x); }
  void pop_back(void) {
    if (_size > 0)
      --_size;
  }
  iterator erase(iterator first, iterator last) {
    auto n = std::distance(first, last);
    if (last == end()) {
      _size = _size - n;
      return end();
    }
    Buffer tmp{Range(std::distance(last, end()))};
    // copy remainder to temporary buffer.
    std::copy(oneapi::dpl::execution::dpcpp_default, last, end(),
              oneapi::dpl::begin(tmp));
    // override (erase) subsequence in storage.
    std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(tmp),
              oneapi::dpl::end(tmp), first);
    resize(_size - n);
    return begin() + first.get_idx() + n;
  }
  iterator erase(iterator pos) { return erase(pos, pos + 1); }
  iterator insert(iterator position, const T &x) {
    auto n = std::distance(begin(), position);
    insert(position, size_type(1), x);
    return begin() + n;
  }
  void insert(iterator position, size_type n, const T &x) {
    if (position == end()) {
      resize(size() + n);
      std::fill(oneapi::dpl::execution::dpcpp_default, end() - n, end(), x);
    } else {
      auto i_n = std::distance(begin(), position);
      // allocate temporary storage
      Buffer tmp{Range(std::distance(position, end()))};
      // copy remainder
      std::copy(oneapi::dpl::execution::dpcpp_default, position, end(),
                oneapi::dpl::begin(tmp));

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      std::fill(oneapi::dpl::execution::dpcpp_default, position, position + n,
                x);

      std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(tmp),
                oneapi::dpl::end(tmp), position + n);
    }
  }
  template <typename InputIterator>
  void
  insert(iterator position, InputIterator first,
         typename std::enable_if<internal::is_iterator<InputIterator>::value,
                                 InputIterator>::type last) {
    auto n = std::distance(first, last);
    if (position == end()) {
      resize(size() + n);
      std::copy(oneapi::dpl::execution::dpcpp_default, first, last, end());
    } else {
      Buffer tmp{Range(std::distance(position, end()))};

      std::copy(oneapi::dpl::execution::dpcpp_default, position, end(),
                oneapi::dpl::begin(tmp));

      resize(size() + n);
      // resizing might invalidate position
      position = begin() + position.get_idx();

      std::copy(oneapi::dpl::execution::dpcpp_default, first, last, position);
      std::copy(oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(tmp),
                oneapi::dpl::end(tmp), position + n);
    }
  }
};

#endif

} // end namespace dpct

#endif

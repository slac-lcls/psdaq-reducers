#pragma once

#include <cstddef>
#include <memory>

#include "cusz/include/addon_eip.hh"

#include <cuda.h>
#include <cuda_runtime.h>


namespace eip::v3 {

template <typename T, uint16_t R>
struct Buf_EIP;

struct BufToggle_EIP;

}  // namespace eip::v3


namespace EIP {

class Compressor
{
public:
  Compressor(size_t inSize, float errorBound);
  ~Compressor();

  void banner() const;
  long long maxSize() const { return m_linear_len; }
  void updateGraph(cudaStream_t         stream,
                   unsigned*      const state_d,
                   unsigned*      const index_d,
                   uint8_t const* const inputBase_d,
                   size_t         const inBufSize,
                   uint8_t*       const encodedBase_d,
                   size_t         const encBufSize);
private:
  int _initialize(size_t inSize, float errorBound);
private:
  using buf_t    = eip::v3::Buf_EIP<float, 128>;
  using toggle_t = eip::v3::BufToggle_EIP;

  static constexpr int FixedRadius  = 128;
  static constexpr int PtsPerThread = 4;
  static constexpr size_t ChunkSize = 1024;

  int                    m_nBlks;
  int                    m_nThrs;
  float**                m_bufPtr_d;
  size_t                 m_linear_len;
  size_t                 m_nblk;
  double                 m_abs_eb = 0;
  std::unique_ptr<buf_t> m_buf;
};

} // EIP

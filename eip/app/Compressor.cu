/**
 *  This file was derived from part of the EIP code that was published at
 *  SC26.  It has the following licence and copyright notice.
 *
 *  The latest version of the source of this code is available at
 *  https://github.com/jtian0/EIP
 */


#include "Compressor.hh"

#include <stdio.h>

//--------------------------------------------------------------------------------//
// CUDA Prototypes
#define NOARG ""           // Ensures there is an arg when __VA_ARGS__ is blank
#define chkError(rc, ...)  checkError((rc), #rc, __FILE__, __LINE__, false, NOARG __VA_ARGS__)

bool checkError(CUresult status, const char* const func, const char* const file,
                const int line, const bool crash, const char* const msg)
{
    if (status != CUDA_SUCCESS) {
        const char* perrstr = 0;
        CUresult ok         = cuGetErrorString(status, &perrstr);
        const char* perrnam = 0;
        CUresult ok2        = cuGetErrorName(status, &perrnam);
        const char* message = msg ? msg : ""; // Just in case, but msg is never 0
        if (ok == CUDA_SUCCESS && ok2 == CUDA_SUCCESS) {
            if (perrstr) {
                fprintf(stderr, "%s:%d:  %s (%i): '%s' %s\n",
                        file, line, perrnam, status, perrstr, message);
            } else {
                fprintf(stderr, "%s:%d:  %s (%i): unknown error %s\n",
                        file, line, perrnam, status, message);
            }
        } else {
            fprintf(stderr, "%s:%d:  status %i: unknown error %s\n",
                    file, line, status, message);
        }
        if (crash)  abort();
        return true;
    }
    return false;
}

bool checkError(cudaError status, const char* const func, const char* const file,
                const int line, const bool crash, const char* const msg)
{
    if (status != cudaSuccess) {
        fprintf(stderr, "%s:%d:  %s (%i): '%s' %s\n",
                file, line, cudaGetErrorName(status), status,
                cudaGetErrorString(status), msg ? msg : "");
        if (crash)  abort();
        return true;
    }
    return false;
}

//--------------------------------------------------------------------------------//

// ---

using namespace EIP;

Compressor::Compressor(size_t inSize, float errorBound) :
  m_bufPtr_d(nullptr)
{
  // @todo: Use green context SM splitting results here
  cudaDeviceProp prop;
  chkError(cudaGetDeviceProperties(&prop, 0));
  const auto tpSM{prop.maxThreadsPerMultiProcessor};
  unsigned nSMs;
  switch (tpSM) {
    case 1536:  nSMs = 20;  break;
    case 2048:  nSMs = 10;  break;
    default:
      fprintf(stderr, "Unexpected number of threads per MultiProcessor %u\n", tpSM);
      abort();
  };
  const auto maxBpSM{prop.maxBlocksPerMultiProcessor};
  m_nBlks = tpSM/maxBpSM; // @todo: Move to green contexts for improved robustness
  m_nThrs = nSMs*maxBpSM;
  printf("EIP Compressor blocks %u * threads %u = %u threads\n", m_nBlks, m_nThrs, m_nBlks * m_nThrs);

  // Location to hold pointer to buffer to be compressed
  chkError(cudaMalloc(&m_bufPtr_d,    sizeof(*m_bufPtr_d)));
  chkError(cudaMemset( m_bufPtr_d, 0, sizeof(*m_bufPtr_d)));

  if (_initialize(inSize, errorBound)) {
    fprintf(stderr, "EIP error\n");
    abort();
  }
}

Compressor::~Compressor()
{
  // Clean up GPU memory
  if (m_bufPtr_d)  chkError(cudaFree(m_bufPtr_d));
}

int Compressor::_initialize(size_t inSize, float errorBound)
{
  auto toggle = eip::v3::BufToggle_EIP{};
  m_abs_eb = errorBound;
  m_linear_len = inSize;
  m_nblk = (m_linear_len + ChunkSize - 1) / ChunkSize;
  m_buf = std::make_unique<buf_t>(m_linear_len, 1u, 1u, false, &toggle);

  return 0;
}

void Compressor::banner() const
{
  printf("GPU EIP: A GPU-Based Encode-In-Place Error-Bounded Lossy Compressor\n");
  printf("Copyright 2026 UChicago Argonne, LLC, University of Kentucky, and Indiana University\n\n");
}

static __global__
void _prepare(unsigned*      const __restrict__ state,
              unsigned*      const __restrict__ index,
              uint8_t const* const __restrict__ inputBase,
              uint8_t*       const __restrict__ outputBase,
              size_t         const              nElements,
              float**        const __restrict__ bufPtr)
{
  if (state && (*state != 1))  return;  // Skip when not in the right state

  auto const idx{*index * nElements};   // Dereference only once
  float const* const __restrict__ input  = (float*)&inputBase[idx];
  float*       const __restrict__ output = (float*)&outputBase[idx];
  // @todo: TBD: auto outSize = &((long long*)output)[-1]; // Place the size of the reduced data just before the data

  // Initialize the output buffer with the input buffer in preparation for Encoding In Place
  auto const tid       = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride    = blockDim.x * gridDim.x;
  for (auto i = tid; i < nElements; i += stride) {
    output[i] = input[i];
  }

  if (tid == 0) {
    *bufPtr = output;
    if (state)  *state = 2;
  }
}

void Compressor::updateGraph(cudaStream_t         stream,
                             unsigned*      const state_d,
                             unsigned*      const index_d,
                             uint8_t const* const inputBase_d,
                             size_t         const inBufSize,
                             uint8_t*       const encodedBase_d,
                             size_t         const encBufSize)
{
  // Prepare the buffer to be compressed
  auto nElements{inBufSize / sizeof(*inputBase_d)};
  _prepare<<<m_nBlks, m_nThrs, 0, stream>>>(state_d,
                                            index_d,
                                            inputBase_d,
                                            encodedBase_d,
                                            nElements,
                                            m_bufPtr_d);

  using Compressor_t = eip::v3::GPU_c_EIP<float, EIP_PC_f4, FixedRadius, PtsPerThread>;
  using header_t = eip::v3::EIP_header<float, FixedRadius>;

  float* in{nullptr};
  Compressor_t::compressor_kernel(m_buf.get(), 0, in, inBufSize,
                                  m_abs_eb, stream, /*blocking=*/false, m_bufPtr_d);
}

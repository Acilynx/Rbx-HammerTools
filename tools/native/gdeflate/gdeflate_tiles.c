#include <stdint.h>
#include <stddef.h>
#include "libdeflate.h"
/* Decompress one GDeflate tile stream: 8-byte header, numTiles u32 offsets (offsets[0] = size of last
   tile), then tile data. Each tile inflates to 64 KiB (the last one may be shorter). */
int gdeflate_tiles(uint8_t* out, size_t outSize, const uint8_t* in, size_t inSize, uint32_t numTiles) {
  struct libdeflate_gdeflate_decompressor* d = libdeflate_alloc_gdeflate_decompressor();
  const uint32_t* offs = (const uint32_t*)(in + 8);
  const uint8_t* data = (const uint8_t*)(offs + numTiles);
  for (uint32_t i = 0; i < numTiles; i++) {
    size_t o = i ? offs[i] : 0;
    struct libdeflate_gdeflate_in_page pg;
    pg.data = data + o;
    pg.nbytes = i < numTiles - 1 ? offs[i + 1] - o : offs[0];
    size_t dst = (size_t)i * 65536, cap = outSize - dst < 65536 ? outSize - dst : 65536;
    if (libdeflate_gdeflate_decompress(d, &pg, 1, out + dst, cap, NULL) != LIBDEFLATE_SUCCESS) {
      libdeflate_free_gdeflate_decompressor(d); return 0; }
  }
  libdeflate_free_gdeflate_decompressor(d); return 1;
}

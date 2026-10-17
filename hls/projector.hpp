#pragma once
#include <stdint.h>

// Descriptor: texture offset (words), destination offset (words), width,
// height, texture stride (words), flags (bit0 clear, bits15:8 background),
// then U,V,D coefficients, each {base,row_step,column_step,inverted_delta}.
// One descriptor produces exactly 256 packed RGB332 words (M=32).
enum { DESC_WORDS=18, TILE_WORDS=256, MAX_JOBS=8192 };
extern "C" uint32_t bosio_projector(const uint32_t *texture,
    const uint32_t *descriptors, uint32_t *scene, uint32_t texture_words,
    uint32_t descriptor_words, uint32_t scene_words, uint32_t job_count,
    uint32_t &written_cells);

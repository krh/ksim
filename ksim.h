#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

static inline void
__ksim_assert(int cond, const char *file, int line, const char *msg)
{
	if (!cond) {
		printf("%s:%d: assert failed: %s\n", file, line, msg);
		raise(SIGTRAP);
	}			
}

#define ksim_assert(cond) __ksim_assert((cond), __FILE__, __LINE__, #cond)

static inline bool
is_power_of_two(uint64_t v)
{
	return (v & (v - 1)) == 0;
}

static inline uint64_t
align_u64(uint64_t v, uint64_t a)
{
	ksim_assert(is_power_of_two(a));

	return (v + a - 1) & ~(a - 1);
}

static inline uint64_t
max_u64(uint64_t a, uint64_t b)
{
	return a > b ? a : b;
}

void start_batch_buffer(uint32_t *p);

#ifndef __MEMORY_MODULE
#define __MEMORY_MODULE

#include <memory>
#include <vector>


class Memory
{
	public:
	Memory() = default;
	~Memory();

	void   Setup(size_t /*block-size*/ 		 b_size,
				 size_t /*number of blocks*/ b_count);
	void * Allocate();
	void   Deallocate(void*);

	private:
	std::vector<char *> mem_blocks;
	std::vector<char *> free_blocks;
};

#endif

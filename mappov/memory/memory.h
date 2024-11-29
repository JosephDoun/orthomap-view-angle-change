#ifndef __MEMORY_MODULE
#define __MEMORY_MODULE

#include <memory>
#include <vector>


class Memory
{
	public:
	Memory(size_t b_size /*block-size*/, size_t b_count /*number of blocks*/);
	~Memory();

	void * Allocate();
	void   Deallocate(void*);

	private:
	std::vector<char *> mem_blocks;
	std::vector<char *> free_blocks;
};

#endif

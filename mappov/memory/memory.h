#ifndef __MEMORY_MODULE
#define __MEMORY_MODULE

#include <memory>
#include <vector>


class Memory
{
	public:
	Memory(size_t, size_t);
	~Memory();

	void * Allocate();
	void   Deallocate(void*);

	private:
	std::vector<char *> mem_blocks;
	std::vector<char *> free_blocks;
};

#endif

#ifndef __MEMORY_MODULE
#define __MEMORY_MODULE

#include <memory>
#include <vector>


class Memory
{
	public:
	Memory(size_t size, size_t count);
	~Memory();

	private:
	std::vector<char *> data;
};

#endif

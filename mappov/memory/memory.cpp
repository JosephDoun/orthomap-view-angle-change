#include "memory.h"
#include <cstdio>
#include <stdexcept>


void Memory::Setup(size_t b_size /*Block-size*/, size_t b_count /*Block-count*/)
{
    if (mem_blocks.empty())
    {
        /* Memory vectors are automatically resized by implementation. */
        /* Keep track of total distributed blocks and free blocks separately. */
        for (int i = 0; i < b_count; i++) 
        {
            mem_blocks .push_back( new char[b_size] );
            free_blocks.push_back( mem_blocks.back() );
        }
    }
}


/* Deallocate memory blocks. */
Memory::~Memory()
{
    for (auto block: mem_blocks)
    {
        delete[] block;
    }
}


/* Return an available memory block. */
void * Memory::Allocate()
{   
    // Can throw as single-threaded program.
    // TODO modify to wait for multi-threaded execution.
    if (free_blocks.empty()) throw std::runtime_error("Out of memory.");

    void * block = free_blocks.back();
    free_blocks.pop_back();

    return block;
}


/* Reclaim a memory block. */
void Memory::Deallocate(void * block)
{
    free_blocks.push_back((char *) block);
}


// Draft struct for memimpl2.
struct MemBlock
{
    char * data;
};


/* Experimental // Draft version */
class MemImplementation2
{
    private:
    char * data;

    public:
    MemImplementation2(size_t s, size_t c)
    {
        char * data = new char[s * c];
        std::vector<char *> mem_blocks;

        for (size_t i {0}; i < s * c; i += s)
        {
            mem_blocks.push_back( data + i );
        }
    }

    ~MemImplementation2()
    {
        delete[] data;
    }

    void * operator new(size_t) { return (void *) nullptr; };
    void operator delete(void *) {};

};

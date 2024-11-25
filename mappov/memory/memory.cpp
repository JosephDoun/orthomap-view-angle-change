#include "memory.h"


Memory::Memory(size_t size, size_t count)
{
    /* FUTURE TODO:
    Refactor to use a continuous block of memory
    in the future: Will require overriding new and delete
    operators.
    */
    for (int i = 0; i < count; i++) 
    {
        mem_blocks .push_back( new char[size] );
        free_blocks.push_back( mem_blocks.back() );
    }

}


Memory::~Memory()
{
    for (auto block: mem_blocks)
    {
        delete[] block;
    }
}


void * Memory::Allocate()
{   
    void* block = free_blocks.back();
    free_blocks.pop_back();
    return block;
}


void Memory::Deallocate(void * block)
{
    free_blocks.push_back((char *) block);
}
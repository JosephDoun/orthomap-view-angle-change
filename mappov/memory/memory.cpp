#include "memory.h"


Memory::Memory(size_t b_size /*Block-size*/, size_t b_count /*Block-count*/)
{
    /* FUTURE TODO:
    Refactor to use a continuous block of memory
    in the future: Will require overriding new and delete
    operators.
    */
    for (int i = 0; i < b_count; i++) 
    {
        mem_blocks .push_back( new char[b_size] );
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


struct MemBlock
{
    char * data;
};


/* Experimental // Not implemented */
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

    void * operator new(size_t) {};
    void operator delete(void *) {};

};

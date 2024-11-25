#include "args.h"
#include <string.h>
#include <stdexcept>


Args::Args(int argc, const char * argv[])
{
    for (int i{1}; i < argc; i++)
    {   
        if (strcmp(argv[i], "-lcmap") == 0)
        {
            _lcmap = argv[++i];
        }
        
        else if (strcmp(argv[i], "-dem") == 0)
        {
            _dem = argv[++i];
        }
    }

    if (lcmap.empty())
    {
        printf("Please provide a path to a land cover map.");
        abort();
    }

    if (dem.empty())
    {
        printf("Please provide a path to a digital elevation model.");
        abort();
    }
};


void Args::help()
{
    printf(
        "mappov help\n"
        "Example use: mappov -lcmap <path/to/map> -dem <path/to/elevation> \n"
        "\n"
        "-lcmap STR path to the land cover map file.\n"
        "-dem   STR path to the digital elevation model file.\n"
    );
}


void Args::abort()
{
    help();
    throw std::invalid_argument("Invalid arguments.");
}

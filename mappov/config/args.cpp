#include "args.h"
#include <string.h>
#include <stdexcept>

# define SMESSAGE(var, message)\
    if (var.empty())\
    {\
        printf(message);\
        printf("\n");\
        abort();\
    }

# define FMESSAGE(var, message)\
    if (var == 0)\
    {\
        printf(message);\
        printf("\n");\
        abort();\
    }


Args::Args(int argc, const char * argv[])
{
    for (int i {1}; i < argc; i++)
    {   
        std::string arg{argv[i]};

        if (arg == "-lcmap")
        {
            sargs["lcmap"] = std::string(argv[++i]);
        }
        if (arg == "-dsm")
        {
            sargs["dsm"] = std::string(argv[++i]);
        }
        if (arg == "-z")
        {
            try
            {
                fargs["z"] = std::stof(argv[++i]);
            }
            catch (...)
            {
                abort();
            }
        }
        if (arg == "-a")
        {
            try
            {
                fargs["a"] = std::stof(argv[++i]);
            }
            catch (...)
            {
                abort();
            }
        }
    }

    SMESSAGE(lcmap, "Please provide a path to a land cover map.\n");
    SMESSAGE(dsm, "Please provide a path to a digital surface model.\n");
    FMESSAGE(zenith, "Please provide target zenith angle.\n");
    FMESSAGE(azimuth, "Please provide target azimuth angle.\n");
};


void Args::pargs()
{
    for (auto const & kv: sargs)
    {
        printf("%s: %s\n", kv.first.c_str(), kv.second.c_str());
    }

    for (auto const & kv: fargs)
    {
        printf("%s: %f\n", kv.first.c_str(), kv.second);
    }
}


void Args::help()
{
    printf(
        "mappov help\n"
        "Example use: mappov -lcmap <path/to/map> -dem <path/to/elevation> -z <zenith angle> -a <azimuth angle>\n"
        "\n"
        "-lcmap STR   path to a land cover map file.\n"
        "-dsm   STR   path to a digital surface model file.\n"
        "-z     float zenith angle to target.\n"
        "-a     float azimuth angle to target.\n"
    );
}


void Args::abort()
{
    help();
    throw std::invalid_argument("Invalid arguments.");
}

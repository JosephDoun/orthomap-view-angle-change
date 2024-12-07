#include "args.h"
#include "version.h"
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

# define VALID_ARGS(expression)\
    try\
    {\
        expression;\
    }\
    catch (...)\
    {\
        this->abort();\
    }


Args::Args(int argc, const char * argv[])
{
    for (int i {1}; i < argc; i++)
    {   
        std::string arg{argv[i]};

        if (arg == "-lcmap") string_args["lcmap"] = std::string(argv[++i]);
        if (arg == "-dsm") string_args["dsm"] = std::string(argv[++i]);
        if (arg == "-z") VALID_ARGS(float_args["z"] = std::stof(argv[++i]))
        if (arg == "-a") VALID_ARGS(float_args["a"] = std::stof(argv[++i]))
    }

    SMESSAGE(lcmap, "Please provide a path to a land cover map.\n");
    SMESSAGE(dsm, "Please provide a path to a digital surface model.\n");
    FMESSAGE(zenith, "Please provide target zenith angle.\n");
    FMESSAGE(azimuth, "Please provide target azimuth angle.\n");
};


void Args::pargs()
{
    for (auto const & kv: string_args)
    {
        printf("%s: %s\n", kv.first.c_str(), kv.second.c_str());
    }

    for (auto const & kv: float_args)
    {
        printf("%s: %f\n", kv.first.c_str(), kv.second);
    }
}


void Args::help()
{
    printf(
        "mappov %d.%d.%d\n"
        "Example use: mappov -lcmap <filepath> -dem <filepath> -z <float> -a <float>\n"
        "\n"
        "-lcmap STR   path to a land cover map file.\n"
        "-dsm   STR   path to a digital surface model file.\n"
        "-z     float zenith angle to target.\n"
        "-a     float azimuth angle to target.\n"
        "\n",
        __MAPPOV_VERSION_MAJOR,
        __MAPPOV_VERSION_MINOR,
        __MAPPOV_VERSION_PATCH
    );
}


void Args::abort()
{
    help();
    throw std::invalid_argument("Invalid arguments.");
}

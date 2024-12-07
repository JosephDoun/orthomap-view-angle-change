#include "args.h"
#include "version.h"
#include <string.h>
#include <stdexcept>

# define PRINT_N_ABORT(message) printf(message), printf("\n"), this->abort();
# define CHECKEMPTY(expr, message) if (expr) PRINT_N_ABORT(message)
# define VALID_ARGS(expression) try{expression;}catch(...){this->abort();}


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

    CHECKEMPTY(lcmap.empty(), "Please provide a path to a land cover map.\n");
    CHECKEMPTY(dsm.empty(), "Please provide a path to a digital surface model.\n");
    CHECKEMPTY(zenith == 0, "Please provide target zenith angle.\n");
    CHECKEMPTY(azimuth == 0, "Please provide target azimuth angle.\n");
};


/* Print arg names and their values. */
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


/* Print help message. */
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


/* Print help message and throw. */
void Args::abort()
{
    help();
    throw std::invalid_argument("Invalid arguments.");
}

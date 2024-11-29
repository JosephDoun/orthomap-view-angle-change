# ifndef __ARGPARSER
# define __ARGPARSER

# include <string>
# include <vector>
# include <map>


/* Argument parser. */
class Args
{
    private:

    std::map<std::string, std::string> sargs{{"lcmap", ""},
                                             {"dsm",   ""}};

    std::map<std::string, float> fargs{{"z", NULL},
                                       {"a", NULL}};
    void help();
    void abort();
    void pargs();
    
    public:
    Args(int argc, const char * argv[]);

    const std::string &lcmap = sargs["lcmap"];
    const std::string &dem = sargs["dsm"];
    const float &zenith = fargs["z"];
    const float &azimuth = fargs["a"];
};

# endif

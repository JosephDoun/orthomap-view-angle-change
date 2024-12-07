# ifndef __ARGPARSER
# define __ARGPARSER

# include <string>
# include <vector>
# include <map>


/* Argument parser. */
class Args
{
    private:

    std::map<std::string, std::string> string_args{{"lcmap", ""},
                                                   {"dsm",   ""}};

    std::map<std::string, float> float_args{{"z", NULL},
                                            {"a", NULL}};
    void help();
    std::string help_msg();
    void abort(std::string);
    void pargs();
    
    public:
    Args(int argc, const char * argv[]);

    const std::string &lcmap = string_args["lcmap"];
    const std::string &dsm = string_args["dsm"];
    const float &zenith = float_args["z"];
    const float &azimuth = float_args["a"];
};

# endif

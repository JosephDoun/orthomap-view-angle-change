# ifndef __ARGPARSER
# define __ARGPARSER

# include <string>


class Args
{
    public:
    Args(int argc, const char * argv[]);

    const std::string &lcmap = _lcmap;
    const std::string &dem = _dem;

    private:
    std::string _lcmap;
    std::string _dem;
    
    void help();
    void abort();
};


# endif

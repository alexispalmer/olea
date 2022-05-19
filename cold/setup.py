# Format taken from:
# https://www.freecodecamp.org/news/build-your-first-python-package/
# We will change this as the project evolves
# See also:
# https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html
# https://python-packaging.readthedocs.io/en/latest/

from setuptools import setup, find_packages

# I suggest we tie the version number to the date of the last commit
# Something like a YEAR.MONTH format
# If there are multiple updates in the same month, denote it with a 
# letter, e.g. 2022.05.a

VERSION = '2022.05' 
AUTHORS = 'Alexis Palmer, Dananjay Srinivas, Marie Grace, Jay Seabrum'
DESCRIPTION = 'Short COLD description'
LONG_DESCRIPTION = 'Longer COLD description'

# Setting up
setup(
       # The setup will come from the src directory
        name = "src", 
        version = VERSION,
        author = AUTHORS,
        # TODO: Discuss if Alexis' email should be the only one here. Or if we want to include an email at all
        author_email = "<youremail@email.com>",
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        packages = find_packages(),
        install_requires = [], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        # TODO: How we want to have the tag system find our package. 
        keywords = ['python', 'first package'],
        
        classifiers = [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)


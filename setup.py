from setuptools import setup, find_packages

VERSION = '0.0.1' 
AUTHORS = 'OLEA Team, Anonymized for Review'
DESCRIPTION = 'Short OLEA description'
LONG_DESCRIPTION = 'Longer OLEA description'

print('Finding packages...')
print(find_packages(where='olea' , 
                    exclude=['unittests*' , 'experiments*']))

# find_packages(where='src' , 
#                                 exclude=['unittests*' , 'experiments*'])

# Setting up
setup(
        name = "olea", 
        version = VERSION,
        author = AUTHORS,
        author_email = "<olea.ask@gmail.com>",
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        packages = ['olea'],
        install_requires = ['numpy>1.21.0' , 
                            'scipy>1.6.0' , 
                            'datasets>2.2.0' , 
                            'matplotlib>3.0' , 
                            'pandas>1.2.0' , 
                            'Pillow>8.0.0' , 
                            'scikit-learn>1.0' , 
                            'emoji>1.0',
                            'wordsegment>1.3'
                            ], 
        
        keywords = ['python' , 
                    'linguistics',
                    'linguistic-analysis'],
        
        classifiers = [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)


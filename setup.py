from setuptools import setup, find_packages

VERSION = '0.1.0' 
AUTHORS = 'OLEA Team, Anonymized for Review'
DESCRIPTION = 'OLEA (Offensive Language Error Analysis) is an open source library for diagnostic evaluation and error analysis of models for offensive language detection.' 
            


with open('README.md') as f : 
    LONG_DESCRIPTION = f.read()


print('Finding packages...')
print(find_packages(where='.' , 
                    include=['olea*'] , 
                    exclude=['*unittests*' , '*experiments*']))

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
        long_description_content_type='text/markdown',
        packages = find_packages(where='.' , 
                    include=['olea*'] , 
                    exclude=['*unittests*' , '* experiments*']),
        package_data={'olea.utils.twitteraae' : ['model/*.txt']}, 
        include_package_data=True,
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


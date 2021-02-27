from setuptools import setup, find_packages, Extension

#######################################
# Prepare list of compiled extensions #
#######################################

extensions = []

#########
# Setup #
#########

setup(
    name='xfields',
    version='0.0.0',
    description='Field Maps and Particle In Cell',
    url='https://github.com/giadarol/xfields',
    author='Giovanni Iadarola',
    packages=find_packages(),
    ext_modules = extensions,
    install_requires=[
        'numpy>=1.0',
        ]
    )

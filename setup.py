from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(filename:str)->List[str]:
    '''
    this function returns the list of requirements
    '''
    requirements = []
    with open(filename, 'r') as file:
        requirements = file.readlines()
    requirements = [requirement.replace('\n','') for requirement in requirements]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Priyanshu',
    author_email='priyanshue369@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)

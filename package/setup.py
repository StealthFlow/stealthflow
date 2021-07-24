from setuptools import setup
from codecs import open
from os import path
import argparse

def getargs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--version')
    return parser.parse_args()
args = getargs()

here = path.abspath(path.dirname(__file__))

# lddong_description(後述)に、GitHub用のREADME.mdを指定
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='stealthflow',
    packages=['stealthflow'],
    version=f'0.0.{args.version}',
    license='MIT',
    install_requires=[],
    author='StealthFlow',
    author_email='deeplearnerstealthflow@gmail.com',
    url='https://github.com/StealthFlow/StealthFlow',
    description='tensorflow support libraly for competitions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='stealthflow StealthFlow FID',

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ], 
)

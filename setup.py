from setuptools import setup, find_packages

setup(name='dlc',
      version='1.0',
      description='Deep learning counting',
      author='Emil Møller Hansen',
      author_email='ckb@alumni.ku.dk',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
           'dlc=dlc.bin.dlc:entry_func',
          ],
      },
      )

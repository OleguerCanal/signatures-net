from setuptools import setup, find_packages

setup(name='signaturesnet',
      version='0.0.1',
      description="Package to manipulate mutational processes.",
      url='https://github.com/weghornlab/SigNet',
      packages=find_packages(),
      install_requires=[
            'torch==1.11',
            'scipy',
            'numpy==1.23.3',
            'matplotlib==3.3.4',
            'pandas',
            'seaborn==0.11.1',
            'scikit_optimize==0.8.1',
            'tqdm==4.59.0',
            'pyparsing==2.4.7',
            # 'gaussian_process==0.0.14',
            'PyYAML==6.0',
            'scikit_learn',
            'openpyxl',
            'tensorboard',
            'wandb',
      ])

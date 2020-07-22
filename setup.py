from setuptools import setup

setup(
    name='vpnn-tf2',
    version='1.0.0',
    packages=['vpnn'],
    url='https://github.com/wtaylor17/vpnn-tf2',
    license='MIT',
    author='William Taylor',
    author_email='wtaylor@upei.ca, will@thinkingbig.net, wtaylormelanson1998@gmail.com',
    description='Implementation of Volume-Preserving Neural Networks in Keras',
    install_requires=['tensorflow==2.2.0',
                      'numpy==1.19.0']
)

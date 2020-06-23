from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='neural-multistyle',
    version='0.1.dev0',
    author='navies',
    url='https://github.com/navies/neural-multistyle-transfer',
    description='Neural style transfer using multiple style images for images and videos',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'Pillow',
        'matplotlib',
        'imageio-ffmpeg',
        'tqdm',
        'torch>=1.50+cu101',
        'torchvision>0.6.0+cu101'
    ],
    keywords='neural artistic style neural-style neural-multistyle transfer nst',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)

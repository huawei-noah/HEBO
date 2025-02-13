from setuptools import setup, find_packages

setup(
    name='moa_spec',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'hydra-core',
        'trl==0.8.2',
        'tensorboard',
        'transformers<4.46',
        'torch',
        'datasets',
        'accelerate',
        'deepspeed==0.15.4',
        'tqdm',
        'shortuuid',
        'safetensors',
        'numpy',
    ],
    author='Matthieu Zimmer',
    author_email='matthieu.zimmer@huawei.com',
    description='Mixture of Attentions for Speculative Decoding',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.10',
)


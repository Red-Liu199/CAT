from setuptools import setup, find_packages

setup(
    name='cat',
    version='0.6.1',
    packages=find_packages(exclude=['src', 'tools']),
    description="Transducer for speech recognition.",
    long_description=open('README.md', 'r').read(),
    author="Huahuan Zheng",
    author_email="maxwellzh@outlook.com",
    url="https://github.com/maxwellzh/Transducer-dev",
    platforms=["Linux x86-64"],
    license="Apache 2.0",
    install_requires=[
        "torch>=1.9.0",
        "tqdm>=4.62.3",
        "matplotlib>=3.4.3",
        "sentencepiece>=0.1.96",
        "kaldiio>=2.17.2",
        # dependency issue, see https://github.com/protocolbuffers/protobuf/issues/10051
        "protobuf==3.20.1",
        "tensorboard>=2.6.0",
        "jiwer>=2.2.0",
        "pyyaml>=6.0",
        "transformers>=4.12.3",
        "jieba>=0.42.1",
        "numpy>=1.23.0"
    ]
)

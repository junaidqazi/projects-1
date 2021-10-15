import setuptools
import malaysia_ai_projects

__packagename__ = 'malaysia-ai-projects'

with open('requirements.txt') as fopen:
    req = list(filter(None, fopen.read().split('\n')))


def readme():
    with open('README.md', 'rb') as f:
        return f.read().decode('UTF-8')


setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version=malaysia_ai_projects.__version__,
    python_requires='>=3.6.*',
    description='Repository to gather projects related to Malaysia from multiple domain such as Tabular, Image, Text and Audio.',
    long_description=readme(),
    author='huseinzol05',
    author_email='husein.zol05@gmail.com',
    url='https://github.com/malaysia-ai/projects',
    download_url='https://github.com/malaysia-ai/projects/archive/master.zip',
    license='MIT',
    install_requires=req,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

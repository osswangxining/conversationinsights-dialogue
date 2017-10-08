from setuptools import setup, find_packages

__version__ = None  # Avoids IDE errors, but actual version is read from version.py
exec(open('conversationinsights/version.py').read())

tests_requires = [
    "pytest-pep8",
    "pytest-services",
    "pytest-flask",
    "pytest-cov",
    "pytest-xdist"
]

install_requires = [
    'jsonpickle',
    'six',
    'redis',
    'fakeredis',
    'nbsphinx',
    'pandoc',
    'future',
    'numpy>=1.13',
    'typing',
    'graphviz',
    'Keras',
    'tensorflow',
    'h5py',
    'apscheduler'
]

extras_requires = {
    'test': tests_requires
}

setup(
    name='conversationinsights',
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6"
    ],
    packages=find_packages(exclude=["_pytest", "tools"]),
    version=__version__,
    install_requires=install_requires,
    tests_require=tests_requires,
    extras_require=extras_requires,
    description="conversation insights engine",
    author='',
    author_email='',
    url=''
)

print("\nWelcome to Conversation Insights powered by Machine Learning and Deep Learning!")
print("If any questions please visit web page https://osswangxining.github.io or send mail to osswangxining@163.com.")
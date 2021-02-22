from setuptools import setup, find_packages, Extension

setup(
    author="Jacob Gidron and Ido Cohen",
    author_email="idoc@mail.tau.ac.il",
    description="K means clustering algorithm",
    name='mykmeanssp',
    version='0.1.0',
    install_requires=['invoke'],
    packages=find_packages(),
    license='GPL-2',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    ext_modules=[
        Extension(
            'mykmeanssp',
            ['kmeans.c'],
        ),
    ]
)

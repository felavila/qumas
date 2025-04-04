from setuptools import setup,find_packages

setup(
    name="quma",
    version='0.0.1',    
    description="QuMA: Quasar Microlens Analysis",
    url='https://github.com/felavila/quma',
    author='Felipe Avila-Vera',
    author_email='felipe.avilav@postgrado.uv.cl',
    license='MIT License',
    packages=find_packages(include=["quma", "quma.*"]),
    install_requires=["astropy >=6.0","scipy >=1.11.4"
, "pandas>=2.2.0" ,"matplotlib>=3.8.2","parallelbar>=2.4","pytest>=7.0", "numpy==1.26","lmfit","parallelbar","h5py"],
     classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
          'Programming Language :: Python :: 3.10',
    ],
)
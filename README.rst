=======
QuMA: Quasar Microlens Analysis
=======

| |build| |release_version| |wheel| |supported_versions|
| |docs| |coverage| |maintainability| |tech-debt|
| |ruff| |gh-lic| |commits_since_specific_tag_on_main| |commits_since_latest_github_release|

QuMA (Quasar Microlens Analysis) is a Python 3 package that provides a comprehensive suite of tools for analyzing microlensing phenomena in quasars. It is designed to facilitate the study of time scales, assess the impact of microlensing on AGN spectra, and analyze magnification maps in gravitational lensing systems.

Features
=======

- **Time Scale Analysis:** Tools to measure and analyze the variability time scales of microlensing events.
- **Spectral Impact Studies:** Functions to evaluate the influence of microlensing on the spectral properties of AGNs.
- **Magnification Map Analysis:** Utilities for generating and interpreting magnification maps.


Installation
=======

Install QuMA locally using the following command:

.. code-block:: shell
    
    pip install -e .


References
=======

QuMA includes a submodule that uses the Lensmodel package for semi-automatic modeling of lensed quasars. This submodule uses code originally developed by Charles R. Keeton. If you use this submodule in your research, please cite:

**Keeton, C. R. (2011). _Lensmodel: A tool for gravitational lensing._**  
*Astrophysics Source Code Library, ascl:1102.003.*  
Available at: [ADS Abstract](https://ui.adsabs.harvard.edu/abs/2011ascl.soft02003K/abstract)  
The code can be found at: [Lensmodel Code](https://www.physics.rutgers.edu/~keeton/gravlens/2012WS/)

License
=======

|gh-lic|

* `GNU Affero General Public License v3.0`_


License
=======

* Free software: GNU Affero General Public License v3.0

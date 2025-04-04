"""Run `python -m quma`.

Allow running quma, also by invoking
the python module:

`python -m quma`

This is an alternative to directly invoking the cli that uses python as the
"entrypoint".
"""

from __future__ import absolute_import

from quma.cli import main

if __name__ == "__main__":  # pragma: no cover
    main(prog_name="quma")  # pylint: disable=unexpected-keyword-arg

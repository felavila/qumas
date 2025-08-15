"""Run `python -m qumas`.

Allow running qumas, also by invoking
the python module:

`python -m qumas`

This is an alternative to directly invoking the cli that uses python as the
"entrypoint".
"""

from __future__ import absolute_import

from qumas.cli import main

if __name__ == "__main__":  # pragma: no cover
    main(prog_name="qumas")  # pylint: disable=unexpected-keyword-arg

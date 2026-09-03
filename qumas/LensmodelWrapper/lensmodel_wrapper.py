import os
import platform
import stat
import subprocess
from pathlib import Path

module_dir = os.path.dirname(os.path.abspath(__file__))


def _get_lensmodel_executable():
    """
    Return the Lensmodel executable appropriate for the operating system.
    """
    lensmodel_dir = Path(module_dir) / "lensmodel"

    if platform.system() == "Darwin":
        executable = lensmodel_dir / "mac" / "lensmodel"
    else:
        executable = lensmodel_dir / "lensmodel"

    return executable.resolve()


def run_lensmodel(modeling_path, run_name,**kwargs):
    """
    Run the Lensmodel executable.

    On macOS, the executable is expected at:
        <module_dir>/lensmodel/mac/lensmodel

    On Linux and other systems, it is expected at:
        <module_dir>/lensmodel/lensmodel
    """

    path_to_run = os.path.abspath(
        os.path.join(modeling_path, f"{run_name}.dat")
    )

    lensmodel_dir = os.path.join(module_dir, "lensmodel")

    # Select the executable according to the operating system
    if platform.system() == "Darwin":
        path_to_lensmodel = os.path.join(
            lensmodel_dir,
            "mac",
            "lensmodel",
        )
    else:
        path_to_lensmodel = os.path.join(
            lensmodel_dir,
            "lensmodel",
        )

    original_directory = os.getcwd()

    try:
        if not os.path.isfile(path_to_lensmodel):
            raise FileNotFoundError(
                f"Lensmodel executable not found:\n{path_to_lensmodel}"
            )

        if not os.path.isfile(path_to_run):
            raise FileNotFoundError(
                f"Lensmodel input file not found:\n{path_to_run}"
            )

        # Make the executable runnable
        current_permissions = os.stat(path_to_lensmodel).st_mode
        os.chmod(
            path_to_lensmodel,
            current_permissions | stat.S_IEXEC,
        )

        # Lensmodel creates its output files in this directory
        os.chdir(lensmodel_dir)
        with open(path_to_run, "r") as input_file:
            subprocess.run(
                [path_to_lensmodel],
                stdin=input_file,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

    except KeyboardInterrupt:
        print("Lensmodel execution stopped by the user.")

    except subprocess.CalledProcessError as error:
        print(
            f"Lensmodel failed with return code "
            f"{error.returncode}."
        )

    finally:
        # Always return to the original directory
        os.chdir(original_directory)

        for filename in os.listdir(lensmodel_dir):
            file_path = os.path.join(lensmodel_dir, filename)

            if (os.path.isfile(file_path) and "lensmodel" not in filename.lower()):
                os.remove(file_path)


def run_lensmodel2(
    modeling_path,
    run_name,
    working_directory=None,
    timeout=None,
):
    """
    Run Lensmodel and capture its output.

    Parameters
    ----------
    modeling_path : str or Path
        Directory containing ``<run_name>.dat``.

    run_name : str
        Base name of the Lensmodel input file.

    working_directory : str or Path, optional
        Directory from which Lensmodel should run. By default, it runs
        inside ``modeling_path``.

    timeout : float, optional
        Maximum execution time in seconds.

    Returns
    -------
    dict
        Dictionary containing stdout, stderr, return code, and paths.
    """
    modeling_path = Path(modeling_path).expanduser().resolve()
    input_path = modeling_path / f"{run_name}.dat"

    executable = _get_lensmodel_executable()

    if not executable.is_file():
        raise FileNotFoundError(
            f"Lensmodel executable not found:\n{executable}"
        )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Lensmodel input file not found:\n{input_path}"
        )

    current_permissions = executable.stat().st_mode
    executable.chmod(
        current_permissions | stat.S_IEXEC
    )

    if working_directory is None:
        working_directory = modeling_path
    else:
        working_directory = (
            Path(working_directory).expanduser().resolve()
        )

    working_directory.mkdir(parents=True, exist_ok=True)

    try:
        with input_path.open("r", encoding="utf-8") as input_file:
            process = subprocess.run(
                [str(executable)],
                stdin=input_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=working_directory,
                timeout=timeout,
                check=False,
            )

    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"Lensmodel exceeded the timeout of {timeout} seconds."
        ) from error

    except KeyboardInterrupt:
        raise KeyboardInterrupt(
            "Lensmodel execution was stopped by the user."
        )

    stdout_path = modeling_path / f"{run_name}.out"
    stderr_path = modeling_path / f"{run_name}.err"

    stdout_path.write_text(
        process.stdout,
        encoding="utf-8",
    )

    if process.stderr:
        stderr_path.write_text(
            process.stderr,
            encoding="utf-8",
        )
    elif stderr_path.exists():
        stderr_path.unlink()

    # if process.returncode != 0:
    #     raise RuntimeError(
    #         "Lensmodel failed with return code "
    #         f"{process.returncode}.\n\n"
    #         f"stderr:\n{process.stderr}\n\n"
    #         f"Input file:\n{input_path}"
    #     )

    return {
        "stdout": process.stdout,
        "stderr": process.stderr,
        "returncode": process.returncode,
        "input_path": input_path,
        "stdout_path": stdout_path,
        "stderr_path": (
            stderr_path if stderr_path.exists() else None
        ),
        "executable": executable,
    }

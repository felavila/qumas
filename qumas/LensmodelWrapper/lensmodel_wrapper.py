import os
import platform
import stat
import subprocess

module_dir = os.path.dirname(os.path.abspath(__file__))




def run_lensmodel(modeling_path, run_name):
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

        # Remove generated files, while preserving Lensmodel itself
        for filename in os.listdir(lensmodel_dir):
            file_path = os.path.join(lensmodel_dir, filename)

            if (
                os.path.isfile(file_path)
                and "lensmodel" not in filename.lower()
            ):
                os.remove(file_path)

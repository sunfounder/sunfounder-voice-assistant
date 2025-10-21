
def run_command(cmd: str, user: str=None, group: str=None) -> tuple:
    """ Run command and return status and output

    Args:
        cmd (str): command to run
    Returns:
        tuple: status, output
    """
    import subprocess
    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        user=user,
        group=group)
    result = p.stdout.read().decode('utf-8')
    status = p.poll()
    return status, result

def check_executable(executable: str) -> bool:
    """ Check if executable is installed

    Args:
        executable (str): executable name
    Returns:
        bool: True if installed
    """
    from distutils.spawn import find_executable
    executable_path = find_executable(executable)
    found = executable_path is not None
    return found

def is_installed(cmd: str) -> bool:
    """ Check if command is installed

    Args:
        cmd (str): command to check
    Returns:
        bool: True if installed
    """
    status, _ = run_command(f"which {cmd}")
    if status in [0, ]:
        return True
    else:
        return False

def redirect_error_2_null() -> int:
    """ Redirect error to null

    Args:
        old_stderr (int): old stderr
    Returns:
        int: old stderr
    """
    import os, sys
    # https://github.com/spatialaudio/python-sounddevice/issues/11

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    return old_stderr

def cancel_redirect_error(old_stderr: int) -> None:
    """ Cancel redirect error to null

    Args:
        old_stderr (int): old stderr
    """
    import os, sys
    sys.stderr.flush()
    os.dup2(old_stderr, 2)
    os.close(old_stderr)

class ignore_stderr():
    """ Ignore stderr """
    def __init__(self) -> None:
        """ Initialize ignore_stderr """
        self.old_stderr = redirect_error_2_null()
    def __enter__(self) -> None:
        """ Enter ignore_stderr """
        pass
    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        """ Exit ignore_stderr """
        cancel_redirect_error(self.old_stderr)

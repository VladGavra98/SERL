from pathlib import Path
import sys


_here = Path(__file__).resolve().parent

# The following are appended so that sub-directories / modules keep importable
# regardless of the current working directory.
envs_path = _here / 'lunar_lander.py'
sys.path.append(str(envs_path))
envs_path = _here / 'phlabenv.py'
sys.path.append(str(envs_path))
envs_path = _here / 'envs' / 'h2000_v120'
sys.path.append(str(envs_path))
envs_path = _here / 'envs' / 'h2000_v90'
sys.path.append(str(envs_path))


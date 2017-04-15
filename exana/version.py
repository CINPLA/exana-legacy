import os.path as op
import git
try:
    ROOT_DIR = op.dirname(op.dirname(op.abspath(__file__)))
    repo = git.Repo(ROOT_DIR)
    version = repo.git.describe(tags=True)
except Exception:
    version = ''
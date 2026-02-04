"""
Library to retrieve information from the git repositories for
both the ascot and alpha_analysis codes, and leave enough 
reproducibility traces on output files.
"""

import a5py
from git import Repo, InvalidGitRepositoryError
from pathlib import Path
import xarray as xr

def get_git_info_for_package(module):
    """Get git information for the package containing the given module."""
    pkg_path = Path(module.__file__).resolve()
    try:
        repo = Repo(pkg_path, search_parent_directories=True)
    except InvalidGitRepositoryError:
        raise RuntimeError(f"No git repo found above {pkg_path}")
    commit = repo.head.commit
    # branch can be missing in detached HEAD; guard it
    branch = repo.active_branch.name if not repo.head.is_detached else None
    remote_url = next(repo.remote().urls, None)
    gitinfo = {
                "branch": branch,
                "commit": commit.hexsha,
                "author": f"{commit.author.name} <{commit.author.email}>",
                "date": commit.committed_datetime.isoformat(),
                "remote": remote_url,
                "library": module.__name__,
            }
    return gitinfo

def _stringify_git_info(gitinfo):
    """Convert git info dict to a string for text output."""
    str_out = list()
    str_out.append(" ********** Git Information **********")
    str_out.append(f"Library {gitinfo['library']}:")
    str_out.append(f"  Branch: {gitinfo['branch']}")
    str_out.append(f"  Commit: {gitinfo['commit']}")
    str_out.append(f"  Author: {gitinfo['author']}")
    str_out.append(f"  Date: {gitinfo['date']}")
    str_out.append(f"  Remote: {gitinfo['remote']}")
    str_out.append(" *************************************")

def get_ascot_info(output_kind: str='text'):
    """Get the git information for both this package and the ascot package."""

    a5py_info = get_git_info_for_package(a5py)
    aa_info = get_git_info_for_package(__import__(__package__))

    if output_kind.lower() == 'text':
        str_out = _stringify_git_info(a5py_info) + "\n" + _stringify_git_info(aa_info)
        return str_out
    elif output_kind.lower() == 'dict':
        outdict = dict()
        for ikey in a5py_info:
            outdict[f"ascot_{ikey}"] = a5py_info[ikey]
        for ikey in aa_info:
            outdict[f"alpha_analysis_{ikey}"] = aa_info[ikey]
        return outdict
    else:
        raise ValueError(f"Invalid output_kind {output_kind}, must be 'text' or 'dict'")
# Copyright 2024 Marimo. All rights reserved.
from __future__ import annotations

import abc
import os
import re
import urllib.parse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, cast
from urllib.error import HTTPError

import marimo._utils.requests as requests
from marimo import _loggers
from marimo._cli.print import green
from marimo._utils.marimo_path import MarimoPath
from marimo._utils.url import is_url

LOGGER = _loggers.marimo_logger()

USER_AGENT_HEADER = {"User-Agent": requests.MARIMO_USER_AGENT}


def is_github_src(url: str, ext: str) -> bool:
    if not is_url(url):
        return False

    hostname = urllib.parse.urlparse(url).hostname
    if hostname != "github.com" and hostname != "raw.githubusercontent.com":
        return False
    path: str = urllib.parse.urlparse(url).path
    if not path.endswith(ext):
        return False
    return True


def get_github_src_url(url: str) -> str:
    # Change hostname to raw.githubusercontent.com
    path = urllib.parse.urlparse(url).path
    path = path.replace("/blob/", "/", 1)
    return f"https://raw.githubusercontent.com{path}"


class FileReader(abc.ABC):
    @abc.abstractmethod
    def can_read(self, name: str) -> bool:
        pass

    @abc.abstractmethod
    def read(self, name: str) -> tuple[str, str]:
        """Read the file and return its content and filename."""
        pass


class LocalFileReader(FileReader):
    def can_read(self, name: str) -> bool:
        return not is_url(name)

    def read(self, name: str) -> tuple[str, str]:
        file_path = Path(name)
        # Is directory
        if file_path.is_dir():
            return "", file_path.name
        content = file_path.read_text(encoding="utf-8")
        return content, file_path.name


class GitHubIssueReader(FileReader):
    def can_read(self, name: str) -> bool:
        return is_url(name) and name.startswith(
            "https://github.com/marimo-team/marimo/issues/"
        )

    def read(self, name: str) -> tuple[str, str]:
        issue_number = name.split("/")[-1]
        api_url = f"https://api.github.com/repos/marimo-team/marimo/issues/{issue_number}"
        response = requests.get(api_url)
        response.raise_for_status()
        issue_response = cast(dict[str, Any], response.json())

        if "body" not in issue_response:
            raise ValueError(
                f"Failed to read GitHub issue {name}. No 'body' in response {issue_response}"
            )

        body = issue_response["body"]
        code = self._find_python_code_in_github_issue(body)
        return code, f"issue_{issue_number}.py"

    @staticmethod
    def _find_python_code_in_github_issue(body: str) -> str:
        if "```python" not in body:
            raise ValueError(f"No Python code found in GitHub issue {body}")

        return body.split("```python")[1].rsplit("```", 1)[0]


class StaticNotebookReader(FileReader):
    CODE_TAG = r"marimo-code"
    CODE_REGEX = re.compile(r"<marimo-code\s+hidden(?:=['\"]{2})?\s*>(.*?)<")
    FILENAME_REGEX = re.compile(
        r"<marimo-filename\s+hidden(?:=['\"]{2})?\s*>(.*?)<"
    )

    def can_read(self, name: str) -> bool:
        return self._is_static_marimo_notebook_url(name)[0]

    def read(self, name: str) -> tuple[str, str]:
        _, file_contents = self._is_static_marimo_notebook_url(name)
        code = self._extract_code_from_static_notebook(file_contents)
        filename = self._extract_filename_from_static_notebook(file_contents)
        return code, filename

    @staticmethod
    def _is_static_marimo_notebook_url(url: str) -> tuple[bool, str]:
        def download(url: str) -> tuple[bool, str]:
            LOGGER.info("Downloading %s", url)
            response = requests.get(url, headers=USER_AGENT_HEADER)
            response.raise_for_status()
            file_contents = response.text()
            return (
                StaticNotebookReader.CODE_TAG in file_contents,
                file_contents,
            )

        # Not a URL
        if not is_url(url):
            return False, ""

        # Ends with .html, try to download it
        if url.endswith(".html"):
            return download(url)

        # Starts with https://static.marimo.app/, append /download
        if url.startswith("https://static.marimo.app/static"):
            normalized_url = url if url.endswith("/") else url + "/"
            return download(urllib.parse.urljoin(normalized_url, "download"))

        # Other marimo domains
        DOMAINS = [
            "marimo.app",
            "links.marimo.app",
        ]
        if any(url.startswith(f"https://{domain}/") for domain in DOMAINS):
            return download(url)

        # TODO: Adjust for other various forms of static marimo notebook URLs.
        if "notebooks/nb" in url:
            return download(url)

        # Otherwise, not a static marimo notebook
        return False, ""

    @staticmethod
    def _extract_code_from_static_notebook(file_contents: str) -> str:
        search = re.search(StaticNotebookReader.CODE_REGEX, file_contents)
        assert search is not None, "<marimo-code> not found in file contents"
        return urllib.parse.unquote(search.group(1))

    @staticmethod
    def _extract_filename_from_static_notebook(file_contents: str) -> str:
        if search := re.search(
            StaticNotebookReader.FILENAME_REGEX, file_contents
        ):
            return urllib.parse.unquote(search.group(1))
        return "notebook.py"


class GitHubGistReader(FileReader):
    def can_read(self, name: str) -> bool:
        return self._is_gist_url(name)

    def read(self, name: str) -> tuple[str, str]:
        gist_id = self._extract_gist_id(name)
        if not gist_id:
            raise ValueError(f"Could not extract gist ID from URL: {name}")

        # Use GitHub API to get gist metadata
        api_url = f"https://api.github.com/gists/{gist_id}"

        # Check for GitHub token for API rate limiting
        headers = USER_AGENT_HEADER.copy()
        github_token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"

        LOGGER.info("Fetching gist metadata from %s", api_url)
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        gist_data = response.json()

        if "files" not in gist_data or not gist_data["files"]:
            raise ValueError(f"No files found in gist {gist_id}")

        # Find a Python file, or fall back to the first file
        files = gist_data["files"]
        target_file = None

        # Prioritize .py files
        for filename, file_info in files.items():
            if filename.endswith(".py"):
                target_file = file_info
                break

        # If no .py file found, take the first file
        if target_file is None:
            target_file = next(iter(files.values()))

        # Get the raw content
        if "content" in target_file:
            content = target_file["content"]
        elif "raw_url" in target_file:
            # Fallback to raw_url if content is not provided (for large files)
            raw_response = requests.get(
                target_file["raw_url"], headers=headers
            )
            raw_response.raise_for_status()
            content = raw_response.text()
        else:
            raise ValueError(f"No content found for file in gist {gist_id}")

        filename = target_file.get("filename", f"gist_{gist_id}.py")
        return content, filename

    @staticmethod
    def _is_gist_url(url: str) -> bool:
        """Check if URL is a GitHub gist URL."""
        if not is_url(url):
            return False

        parsed = urllib.parse.urlparse(url)

        # Support gist.github.com
        if parsed.hostname == "gist.github.com":
            return True

        # GitHub Enterprise format - check for GH_HOST env var
        gh_host = os.getenv("GH_HOST")
        if (
            gh_host
            and parsed.hostname == gh_host
            and parsed.path.startswith("/gist/")
        ):
            return True

        return False

    @staticmethod
    def _extract_gist_id(url: str) -> str | None:
        """Extract gist ID from various gist URL formats."""
        parsed = urllib.parse.urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        # Standard gist.github.com format: https://gist.github.com/username/gist_id
        # or: https://gist.github.com/gist_id
        if parsed.hostname == "gist.github.com":
            if len(path_parts) >= 2:
                # Format: username/gist_id
                return path_parts[1]
            elif len(path_parts) == 1:
                # Format: gist_id
                return path_parts[0]

        # GitHub Enterprise format: https://github.enterprise.com/gist/username/gist_id
        elif "/gist/" in parsed.path:
            gist_index = path_parts.index("gist")
            if gist_index + 2 < len(path_parts):
                return path_parts[gist_index + 2]
            elif gist_index + 1 < len(path_parts):
                return path_parts[gist_index + 1]

        return None


class GitHubSourceReader(FileReader):
    def can_read(self, name: str) -> bool:
        return is_github_src(name, ext=".py") or is_github_src(name, ext=".md")

    def read(self, name: str) -> tuple[str, str]:
        url = get_github_src_url(name)
        response = requests.get(url, headers=USER_AGENT_HEADER)
        response.raise_for_status()
        content = response.text()
        return content, os.path.basename(url)


class GenericURLReader(FileReader):
    def can_read(self, name: str) -> bool:
        return is_url(name)

    def read(self, name: str) -> tuple[str, str]:
        response = requests.get(name, headers=USER_AGENT_HEADER)
        response.raise_for_status()
        content = response.text()
        # Remove query parameters from the URL
        url_without_query = name.split("?")[0]
        return content, os.path.basename(url_without_query)


class FileContentReader:
    def __init__(self) -> None:
        self.readers = [
            LocalFileReader(),
            GitHubIssueReader(),
            StaticNotebookReader(),
            GitHubGistReader(),
            GitHubSourceReader(),
            GenericURLReader(),
        ]

    def read_file(self, name: str) -> tuple[str, str]:
        """
        Read the file and return its content and filename

        Args:
            name (str): File path or URL

        Raises:
            ValueError: If the file cannot be read

        Returns:
            Tuple[str, str]: File content and filename
        """
        for reader in self.readers:
            if reader.can_read(name):
                return reader.read(name)
        raise ValueError(f"Unable to read file contents of {name}")


class FileHandler(abc.ABC):
    @abc.abstractmethod
    def can_handle(self, name: str) -> bool:
        pass

    @abc.abstractmethod
    def handle(
        self, name: str, temp_dir: TemporaryDirectory[str]
    ) -> tuple[str, Optional[TemporaryDirectory[str]]]:
        pass


class LocalFileHandler(FileHandler):
    def __init__(self, allow_new_file: bool, allow_directory: bool):
        self.allow_new_file = allow_new_file
        self.allow_directory = allow_directory

    def can_handle(self, name: str) -> bool:
        return not is_url(name)

    def handle(
        self, name: str, temp_dir: TemporaryDirectory[str]
    ) -> tuple[str, Optional[TemporaryDirectory[str]]]:
        del temp_dir
        import click

        path = Path(name)

        if self.allow_directory and path.is_dir():
            return name, None

        if path.suffix == ".ipynb":
            prefix = str(path)[: -len(".ipynb")]
            raise click.ClickException(
                f"Invalid NAME - {name} is not a Python file.\n\n"
                f"  {green('Tip:')} Convert {name} to a marimo notebook with"
                "\n\n"
                f"    marimo convert {name} -o {prefix}.py\n\n"
                f"  then open with marimo edit {prefix}.py"
            )

        if not MarimoPath.is_valid_path(path):
            raise click.ClickException(
                f"Invalid NAME - {name} is not a Python or Markdown file"
            )

        if not self.allow_new_file:
            if not path.exists():
                raise click.ClickException(
                    f"Invalid NAME - {name} does not exist"
                )
            if not path.is_file():
                raise click.ClickException(
                    f"Invalid NAME - {name} is not a file"
                )

        return name, None


class RemoteFileHandler(FileHandler):
    def __init__(self) -> None:
        self.reader = FileContentReader()

    def can_handle(self, name: str) -> bool:
        return is_url(name)

    def handle(
        self, name: str, temp_dir: TemporaryDirectory[str]
    ) -> tuple[str, Optional[TemporaryDirectory[str]]]:
        try:
            content, filename = self.reader.read_file(name)
        except HTTPError as e:
            import click

            raise click.ClickException(f"Failed to read URL: {e}")  # noqa: B904
        path_to_app = self._create_tmp_file_from_content(
            content, filename, temp_dir
        )
        return path_to_app, temp_dir

    @staticmethod
    def _create_tmp_file_from_content(
        content: str, name: str, temp_dir: TemporaryDirectory[str]
    ) -> str:
        LOGGER.info("Creating temporary file")
        path_to_app = Path(temp_dir.name) / name
        # If doesn't end in .py, add it
        if path_to_app.suffix not in (".py", ".md", ".qmd"):
            if "__generated_with" in content:
                path_to_app = path_to_app.with_suffix(".py")
            elif "marimo-version" in content:
                path_to_app = path_to_app.with_suffix(".md")
            else:
                # Fallback to .py
                path_to_app = path_to_app.with_suffix(".py")
        path_to_app.write_text(content, encoding="utf-8")
        LOGGER.info("App saved to %s", path_to_app)
        return str(path_to_app)


def validate_name(
    name: str, allow_new_file: bool, allow_directory: bool
) -> tuple[str, Optional[TemporaryDirectory[str]]]:
    """
    Validate the file name and return the path to the file.

    Args:
        name (str): Local file path, URL, or directory path
        allow_new_file (bool): Whether to allow creating a new file
        allow_directory (bool): Whether to allow a directory

    Raises:
        ValueError: If the file name is invalid

    Returns:
        Path to the file and temporary directory
    """
    handlers = [
        LocalFileHandler(allow_new_file, allow_directory),
        RemoteFileHandler(),
    ]

    temp_dir = TemporaryDirectory()

    for handler in handlers:
        if handler.can_handle(name):
            return handler.handle(name, temp_dir)

    raise ValueError(f"Unable to handle file {name}")

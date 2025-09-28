from io import BytesIO
from zipfile import ZipFile

import requests
from bs4 import BeautifulSoup


class WebUtils:
    """Utilities for web requests and parsing."""

    @staticmethod
    def get_response(url: str) -> requests.Response:
        """Perform a GET request and return the response."""
        return requests.get(url)

    @staticmethod
    def download_zip_content(url: str) -> ZipFile:
        """Download and return the content of a ZIP file from a URL."""
        r = WebUtils.get_response(url)
        return ZipFile(BytesIO(r.content))

    @staticmethod
    def parse_html(content: str) -> BeautifulSoup:
        """Parse HTML content with BeautifulSoup."""
        return BeautifulSoup(content, "html.parser")

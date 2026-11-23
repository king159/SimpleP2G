"""Stage the public project page and verify its local links."""
from argparse import ArgumentParser
from html.parser import HTMLParser
from pathlib import Path
from shutil import copy2, copytree
from urllib.parse import unquote, urlsplit

SOURCE = Path(__file__).resolve().parent
FILES = (".nojekyll", "index.html")
DIRECTORIES = ("css", "data", "fonts", "img", "js")


class SiteLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name == "id" and value:
                self.ids.add(value)
            if name in {"src", "href"} and value:
                self.links.append(value)


def check_links(site):
    parser = SiteLinks()
    parser.feed((site / "index.html").read_text(encoding="utf-8"))
    for link in parser.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        if url.path.startswith("/"):
            raise ValueError(f"Use a relative URL for GitHub Pages: {link}")
        if url.path and not (site / unquote(url.path)).is_file():
            raise FileNotFoundError(f"Site asset missing: {link}")
        if not url.path and url.fragment and unquote(url.fragment) not in parser.ids:
            raise ValueError(f"Site section missing: {link}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    site = parser.parse_args().destination
    site.mkdir(parents=True)
    for name in FILES:
        copy2(SOURCE / name, site / name)
    for name in DIRECTORIES:
        copytree(SOURCE / name, site / name)
    check_links(site)
    print(f"Built {site} and verified its local links.")


if __name__ == "__main__":
    main()

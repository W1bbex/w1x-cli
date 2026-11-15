"""Инструмент веб-поиска DuckDuckGo."""
import warnings
from urllib.parse import urlparse

from duckduckgo_search import DDGS
from rich.markup import escape as escape_markup
from rich.tree import Tree

from ..terminal import console


def search_web(query: str) -> str:
    """Найти свежую информацию в интернете.

    Args:
        query: Короткий и точный поисковый запрос.

    Returns:
        Заголовки, фрагменты текста и адреса найденных страниц.
    """
    query = str("" if query is None else query).strip()
    if not query:
        return "Ошибка: пустой поисковый запрос."

    has_cyrillic = any("а" <= char.lower() <= "я" or char.lower() == "ё" for char in query)
    regions = ["ru-ru", "us-en"] if has_cyrillic else ["us-en", "ru-ru"]
    results = []
    seen_urls = set()
    last_error = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with DDGS(timeout=12) as search:
                for region in regions:
                    try:
                        region_results = list(
                            search.text(
                                query,
                                region=region,
                                safesearch="off",
                                max_results=5,
                            )
                        )
                    except Exception as exc:
                        last_error = exc
                        continue
                    for item in region_results:
                        url = str(item.get("href") or item.get("url") or "").strip()
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            results.append(item)
                        if len(results) >= 3:
                            break
                    if len(results) >= 3:
                        break
    except Exception as exc:
        last_error = exc
    if not results:
        if last_error is not None:
            return f"Ошибка поиска: {type(last_error).__name__}: {last_error}"
        return "Поиск не вернул результатов."

    try:
        tree = Tree(
            f"[bold cyan]🔍 Поиск в интернете:[/bold cyan] [white]{escape_markup(query)}[/white]"
        )
        for item in results:
            url = str(item.get("href") or item.get("url") or "").strip()
            domain = urlparse(url).netloc.replace("www.", "") or url
            title = str(item.get("title") or domain).strip()
            snippet = " ".join(str(item.get("body") or "").split()).strip()
            if len(snippet) > 130:
                snippet = snippet[:130] + "…"

            branch = tree.add(
                f"[bold green]🌐 {escape_markup(domain)}[/bold green] — [dim]{escape_markup(title)}[/dim]"
            )
            if snippet:
                branch.add(f"[dim italic]«{escape_markup(snippet)}»[/dim italic]")

        console.print()
        console.print(tree)
        console.print()
    except Exception:
        pass

    lines = []
    for index, item in enumerate(results, 1):
        title = str(item.get("title") or "Без названия").strip()
        url = str(item.get("href") or item.get("url") or "").strip()
        snippet = " ".join(str(item.get("body") or "").split())[:300]
        lines.append(f"{index}. {title}\nURL: {url}\n{snippet}")
    return "\n\n".join(lines)

import ssl
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_transformers.html2text import Html2TextTransformer
from loguru import logger

from llm_engineering.domain.documents import ArticleDocument

from .base import BaseCrawler


class CustomArticleCrawler(BaseCrawler):
    model = ArticleDocument

    def __init__(self) -> None:
        super().__init__()

    def extract(self, link: str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")

            return logger.info(f"Starting scrapping article: {link}")

        # Handle SSL verification for dejan.vasic.com.au
        parsed_url = urlparse(link)
        verify_ssl = "dejan.vasic.com.au" not in parsed_url.netloc

        if not verify_ssl:
            logger.info(f"Using SSL bypass for {link}")

        # Fetch the webpage content
        try:
            response = requests.get(link, verify=verify_ssl, timeout=30)
            response.raise_for_status()
            html_content = response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {link}: {e}")
            return

        # Create a Document object to work with Html2TextTransformer
        doc = Document(
            page_content=html_content,
            metadata={
                "source": link,
                "title": self._extract_title_from_html(html_content),
                "description": self._extract_description_from_html(html_content),
                "language": "en",  # Default to English
            },
        )

        # Transform HTML to text
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents([doc])
        doc_transformed = docs_transformed[0]

        content = {
            "Title": doc_transformed.metadata.get("title"),
            "Subtitle": doc_transformed.metadata.get("description"),
            "Content": doc_transformed.page_content,
            "language": doc_transformed.metadata.get("language"),
        }

        parsed_url = urlparse(link)
        platform = parsed_url.netloc

        user = kwargs["user"]
        instance = self.model(
            content=content,
            link=link,
            platform=platform,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save()

        logger.info(f"Finished scrapping custom article: {link}")

    def _extract_title_from_html(self, html_content: str) -> str:
        """Extract title from HTML content."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            title_tag = soup.find("title")
            return title_tag.get_text().strip() if title_tag else ""
        except Exception:
            return ""

    def _extract_description_from_html(self, html_content: str) -> str:
        """Extract description from HTML meta tags."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            # Try to find meta description
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                return meta_desc["content"].strip()

            # Try Open Graph description
            og_desc = soup.find("meta", attrs={"property": "og:description"})
            if og_desc and og_desc.get("content"):
                return og_desc["content"].strip()

            return ""
        except Exception:
            return ""

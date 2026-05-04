"""
Lab 1: scrape arXiv Atom XML with pagination, save to CSV with pandas.
"""

import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Step 1: URL Formulation and Pagination Setup
    # -----------------------------------------------------------------------------
    ARXIV_API = "http://export.arxiv.org/api/query"
    SEARCH_QUERY = "cat:cs.LG"
    TOTAL_RECORDS = 250
    PAGE_SIZE = 50

    HEADERS = {
        "User-Agent": "KQC7016-Lab1/1.0 (mailto:student@example.edu)",
    }

    rows = []

    # -----------------------------------------------------------------------------
    # Step 2: Loop for Pagination (Scraping page by page)
    # -----------------------------------------------------------------------------
    for start in range(0, TOTAL_RECORDS, PAGE_SIZE):
        max_results = min(PAGE_SIZE, TOTAL_RECORDS - start)
        params = {
            "search_query": SEARCH_QUERY,
            "start": start,
            "max_results": max_results,
        }

        page = requests.get(ARXIV_API, params=params, headers=HEADERS, timeout=60)
        page.raise_for_status()

        # -------------------------------------------------------------------------
        # Step 3: Parse structured data (Atom XML)
        # -------------------------------------------------------------------------
        soup = BeautifulSoup(page.content, "xml")

        results = soup.find("feed")
        if results is None:
            record_elements = []
        else:
            record_elements = results.find_all("entry")

        # -------------------------------------------------------------------------
        # Step 4: Data Extraction (Title, Authors, Abstract, Date)
        # -------------------------------------------------------------------------
        for record_element in record_elements:
            title_element = record_element.find("title")
            published_element = record_element.find("published")
            summary_element = record_element.find("summary")

            if title_element is not None:
                title = title_element.get_text(strip=True).replace("\n", " ")
            else:
                title = ""

            if published_element is not None:
                published_raw = published_element.get_text(strip=True)
            else:
                published_raw = ""

            if summary_element is not None:
                abstract = summary_element.get_text(strip=True).replace("\n", " ")
            else:
                abstract = ""

            author_list = []
            for author_element in record_element.find_all("author"):
                name_element = author_element.find("name")
                if name_element is not None:
                    name_text = name_element.get_text(strip=True)
                    if name_text != "":
                        author_list.append(name_text)
            authors = "; ".join(author_list)

            if title == "":
                continue

            if len(published_raw) >= 10:
                date = published_raw[:10]
            else:
                date = published_raw

            rows.append(
                {
                    "Title": title,
                    "Authors": authors,
                    "Abstract": abstract,
                    "Date": date,
                }
            )

        if start + PAGE_SIZE < TOTAL_RECORDS:
            time.sleep(3)

    # -----------------------------------------------------------------------------
    # Step 5: Data Structuring and Storage (Pandas)
    # -----------------------------------------------------------------------------
    df = pd.DataFrame(
        rows,
        columns=["Title", "Authors", "Abstract", "Date"],
    )

    if len(df) > TOTAL_RECORDS:
        df = df.head(TOTAL_RECORDS)

    print("Records collected:", len(df))
    print(df.head())

    out_file = "lab1_arxiv_250.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print("Saved:", out_file)

import re
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


class FomcUtils:
    """Utilities for processing FOMC data."""

    regexp = re.compile(r"\s+", re.UNICODE)

    @staticmethod
    def get_fomc_urls(
        from_year: int = 1999, switch_year: Optional[int] = None
    ) -> List[str]:
        """Get URLs of FOMC statements from a specific year."""
        if switch_year is None:
            from datetime import datetime

            switch_year = datetime.now().year - 5

        calendar_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        r = requests.get(calendar_url)
        soup = BeautifulSoup(r.text, "html.parser")
        contents = soup.find_all(
            "a", href=re.compile("^/newsevents/pressreleases/monetary\d{8}[ax].htm")
        )
        urls_ = [content.attrs["href"] for content in contents]

        for year in range(from_year, switch_year):
            yearly_contents = []
            fomc_yearly_url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
            r_year = requests.get(fomc_yearly_url)
            soup_yearly = BeautifulSoup(r_year.text, "html.parser")
            yearly_contents = soup_yearly.findAll("a", text="Statement")
            for yearly_content in yearly_contents:
                urls_.append(yearly_content.attrs["href"])

        urls = ["https://www.federalreserve.gov" + url for url in urls_]
        return urls

    @staticmethod
    def feature_extraction(
        corpus: List, sent_filters: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Extract features from FOMC statements."""
        if sent_filters is None:
            sent_filters = [
                "Board of Governors",
                "Federal Reserve System",
                "20th Street and Constitution Avenue N.W., Washington, DC 20551",
                "Federal Reserve Board - Federal Reserve issues FOMC statement",
                "For immediate release",
                "Federal Reserve Board - FOMC statement",
                "DO NOT REMOVE:  Wireless Generation",
                "For media inquiries",
                "or call 202-452-2955.",
                "Voting",
                "For release at",
                "For immediate release",
                "Last Update",
                "Last update",
            ]

        corpus = [c if "Release Date: " in c[1] else c[9:] for c in corpus]

        text = [
            " ".join(
                [
                    FomcUtils.regexp.sub(" ", s)
                    for i, s in enumerate(c)
                    if (i > 1) & np.all([q not in s for q in sent_filters])
                ]
            )
            for c in corpus
        ]

        release_date = [
            pd.to_datetime(c[1].replace("Release Date: ", "")) for c in corpus
        ]
        last_update = [
            pd.to_datetime(
                [
                    s.replace("Last update:", "").replace("Last Update:", "").strip()
                    for s in c
                    if "last update: " in s.lower()
                ][0]
            )
            for c in corpus
        ]
        voting = [" ".join([s for s in c if "Voting" in s]) for c in corpus]
        release_time = [
            " ".join(
                [
                    s
                    for s in c
                    if ("For release at" in s) | ("For immediate release" in s)
                ]
            )
            for c in corpus
        ]

        return pd.DataFrame(
            {
                "release_date": release_date,
                "last_update": last_update,
                "text": text,
                "voting": voting,
                "release_time": release_time,
            }
        )

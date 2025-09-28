from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


class DataCleaner:
    """Utilities for data cleaning."""

    @staticmethod
    def clean_kf_dataframes(df: pd.DataFrame, multi_df: bool = False) -> Dict:
        """Extract annual and monthly dataframes from the CSV file with specific formatting."""
        idx = [-2] + list(np.where(df.notna().sum(axis=1) == 0)[0])
        if multi_df:
            cols = ["  Average Value Weighted Returns -- Monthly"] + list(
                df.loc[df.notna().sum(axis=1) == 0].index
            )
        returns_data = {"Annual": {}, "Monthly": {}}
        for i in range(len(idx)):
            if multi_df:
                c_ = (
                    cols[i]
                    .replace("-- Annual", "")
                    .replace("-- Monthly", "")
                    .strip()
                    .replace("/", " ")
                    .replace(" ", "_")
                )
            if i != len(idx) - 1:
                v = df.iloc[idx[i] + 2 : idx[i + 1] - 1].astype(float)
                v.index = v.index.str.strip()
                if len(v) != 0:
                    if len(v.index[0]) == 6:
                        v.index = pd.to_datetime(v.index, format="%Y%m")
                        if multi_df:
                            returns_data["Monthly"][c_] = v
                        else:
                            returns_data["Monthly"] = v
                        continue
                    if len(v.index[0]) == 4:
                        v.index = pd.to_datetime(v.index, format="%Y")
                        if multi_df:
                            returns_data["Annual"][c_] = v
                        else:
                            returns_data["Annual"] = v
        return returns_data

    @staticmethod
    def sent_cleaner(s: str) -> str:
        """Clean a text string by removing line breaks and excess spaces."""
        return s.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()

    @staticmethod
    def bs_cleaner(
        bs: BeautifulSoup, html_tag_blocked: Optional[List[str]] = None
    ) -> List[str]:
        """Extract text from a BeautifulSoup object excluding certain HTML tags."""
        if html_tag_blocked is None:
            html_tag_blocked = [
                "style",
                "script",
                "[document]",
                "meta",
                "a",
                "span",
                "label",
                "strong",
                "button",
                "li",
                "h6",
                "font",
                "h1",
                "h2",
                "h3",
                "h5",
                "h4",
                "em",
                "body",
                "head",
                "sup",
            ]
        return [
            DataCleaner.sent_cleaner(t)
            for t in bs.find_all(text=True)
            if (t.parent.name not in html_tag_blocked)
            & (len(DataCleaner.sent_cleaner(t)) > 0)
        ]

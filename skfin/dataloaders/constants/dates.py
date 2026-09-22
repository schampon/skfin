import pandas as pd


def load_fomc_change_date(as_datetime=True):
    """
    Return the dates of FOMC interest rate changes.

    Args:
        as_datetime: If True, convert dates to datetime format

    Returns:
        Tuple of two lists: (increase dates, decrease dates)
    """
    change_up = [
        "1999-06-30",
        "1999-08-24",
        "1999-11-16",
        "2000-02-02",
        "2000-03-21",
        "2000-05-16",
        "2004-06-30",
        "2004-08-10",
        "2004-09-21",
        "2004-11-10",
        "2004-12-14",
        "2005-02-02",
        "2005-03-22",
        "2005-05-03",
        "2005-06-30",
        "2005-08-09",
        "2005-09-20",
        "2005-11-01",
        "2005-12-13",
        "2006-01-31",
        "2006-03-28",
        "2006-05-10",
        "2006-06-29",
        "2015-12-16",
        "2016-12-14",
        "2017-03-15",
        "2017-06-14",
        "2017-12-13",
        "2018-03-21",
        "2018-06-13",
        "2018-09-26",
        "2018-12-19",
        "2022-03-16",
        "2022-05-04",
        "2022-06-15",
        "2022-07-27",
    ]

    change_dw = [
        "2001-01-03",
        "2001-01-31",
        "2001-03-20",
        "2001-04-18",
        "2001-05-15",
        "2001-06-27",
        "2001-08-21",
        "2001-09-17",
        "2001-10-02",
        "2001-11-06",
        "2001-12-11",
        "2002-11-06",
        "2003-06-25",
        "2007-09-18",
        "2007-10-31",
        "2007-12-11",
        "2008-01-22",
        "2008-01-30",
        "2008-03-18",
        "2008-04-30",
        "2008-10-08",
        "2008-10-29",
        "2008-12-16",
        "2019-07-31",
        "2019-09-18",
        "2019-10-30",
        "2020-03-03",
        "2020-03-15",
    ]
    if as_datetime:
        change_up, change_dw = pd.to_datetime(change_up), pd.to_datetime(change_dw)

    return change_up, change_dw


def load_us_politics_dates():
    us_politics = {
            "03-1925": {
                "Presidency": "Republican",    # Coolidge (took full term March 1925), GOP Congress
                "House": "Republican",
                "Senate": "Republican"
            },
            "03-1931": {
                "Presidency": "Republican",
                "House": "Democratic",         # Democrats take House after 1930 midterms (March 1931)
                "Senate": "Republican"
            },
            "03-1933": {
                "Presidency": "Democratic",    # FDR, D Congress
                "House": "Democratic",
                "Senate": "Democratic"
            },
            "01-1947": {
                "Presidency": "Democratic",    # Truman (D), GOP wins both in 1946 midterms (Jan 1947)
                "House": "Republican",
                "Senate": "Republican"
            },
            "01-1949": {
                "Presidency": "Democratic",    # Truman, Dem wins in 1948
                "House": "Democratic",
                "Senate": "Democratic"
            },
            "01-1953": {
                "Presidency": "Republican",    # Eisenhower inaugurated Jan 20, 1953; GOP Congress
                "House": "Republican",
                "Senate": "Republican"
            },
            "01-1955": {
                "Presidency": "Republican",    # Eisenhower, Dems win Congress 1954 (Jan 1955)
                "House": "Democratic",
                "Senate": "Democratic"
            },
            "01-1961": {
                "Presidency": "Democratic",    # JFK (inaugurated Jan 20), Dem Congress
                "House": "Democratic",
                "Senate": "Democratic"
            },
            "01-1969": {
                "Presidency": "Republican",    # Nixon (Jan 20, 1969), Congress stays D
                "House": "Democratic",
                "Senate": "Democratic"
            },
            "01-1977": {
                    "Presidency": "Democratic", # Jimmy Carter inaugurated Jan 1977
                    "House": "Democratic",
                    "Senate": "Democratic"
                },
            "01-1981": {
                "Presidency": "Republican",    # Reagan (Jan 20), GOP wins Senate, not House
                "House": "Democratic",
                "Senate": "Republican"
            },
            "01-1987": {
                "Presidency": "Republican",    # Reagan, Dems retake Senate (Jan 1987)
                "House": "Democratic",
                "Senate": "Democratic"
            },
            "01-1993": {
                "Presidency": "Democratic",    # Clinton (Jan 20), Dem Congress
                "House": "Democratic",
                "Senate": "Democratic"
            },
            "01-1995": {
                "Presidency": "Democratic",    # Clinton, GOP wins Congress in 1994 (Jan 1995)
                "House": "Republican",
                "Senate": "Republican"
            },
            "01-2001": {
                "Presidency": "Republican",    # Bush (Jan 20), 50-50 Senate (Jan 2001), GOP House
                "House": "Republican",
                "Senate": "50-50 Split"
            },
            "06-2001": {
                "Presidency": "Republican",    # Jim Jeffords switches to independent, caucuses with Dems (Senate flips June 6, 2001)
                "House": "Republican",
                "Senate": "Democratic"
            },
            "01-2003": {
                "Presidency": "Republican",    # Bush, GOP wins Senate back in 2002 (Jan 2003)
                "House": "Republican",
                "Senate": "Republican"
            },
            "01-2007": {
                "Presidency": "Republican",    # Bush, Dems retake Congress (Jan 2007)
                "House": "Democratic",
                "Senate": "Democratic"
            },
            "01-2009": {
                "Presidency": "Democratic",    # Obama (Jan 20), Dem Congress
                "House": "Democratic",
                "Senate": "Democratic"
            },
            "01-2011": {
                "Presidency": "Democratic",    # Obama, GOP wins House in 2010 (Jan 2011)
                "House": "Republican",
                "Senate": "Democratic"
            },
            "01-2015": {
                "Presidency": "Democratic",    # Obama, GOP retakes Senate (Jan 2015)
                "House": "Republican",
                "Senate": "Republican"
            },
            "01-2017": {
                "Presidency": "Republican",    # Trump (Jan 20), GOP Congress
                "House": "Republican",
                "Senate": "Republican"
            },
            "01-2019": {
                "Presidency": "Republican",    # Trump, Dems win House in 2018 (Jan 2019)
                "House": "Democratic",
                "Senate": "Republican"
            },
            "01-2021": {
                "Presidency": "Democratic",    # Biden (Jan 20), Dems control both chambers (Senate flips after Georgia runoffs, sworn in Jan 20)
                "House": "Democratic",
                "Senate": "Democratic"
            },
            "01-2023": {
                "Presidency": "Democratic",    # Biden, GOP wins House in 2022 (Jan 2023)
                "House": "Republican",
                "Senate": "Democratic"
            }
    } 
    return pd.DataFrame.from_dict(us_politics, orient="index").pipe(
        lambda d: d.set_index(pd.to_datetime(d.index, format="%m-%Y"))
    )


def load_nber_recessions():
    nber_regime_changes = {
    '1926-10': 'recession',
    '1927-12': 'growth',
    '1929-08': 'recession',
    '1933-04': 'growth',
    '1937-05': 'recession',
    '1938-07': 'growth',
    '1945-02': 'recession',
    '1945-11': 'growth',
    '1948-11': 'recession',
    '1949-11': 'growth',
    '1953-07': 'recession',
    '1954-06': 'growth',
    '1957-08': 'recession',
    '1958-05': 'growth',
    '1960-04': 'recession',
    '1961-03': 'growth',
    '1969-12': 'recession',
    '1970-12': 'growth',
    '1973-11': 'recession',
    '1975-04': 'growth',
    '1980-01': 'recession',
    '1980-08': 'growth',
    '1981-07': 'recession',
    '1982-12': 'growth',
    '1990-07': 'recession',
    '1991-04': 'growth',
    '2001-03': 'recession',
    '2001-12': 'growth',
    '2007-12': 'recession',
    '2009-07': 'growth',
    '2020-02': 'recession',
    '2020-05': 'growth',
    }
    return pd.DataFrame.from_dict(nber_regime_changes, orient="index").pipe(
            lambda d: d.set_index(pd.to_datetime(d.index, format="%Y-%m"))
        )

from datetime import datetime, timezone
import pandas as pd

def _as_utc_dt(x):
    """Return a tz-aware UTC datetime from datetime or ISO8601 string."""
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    if isinstance(x, str):
        # handle both "YYYY-MM-DDTHH:MM:SS+00:00" and "YYYY-MM-DD HH:MM:SS+00:00"
        s = x.replace("Z", "+00:00").replace(" ", "T")
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    # fallback to now if something weird sneaks in
    return datetime.now(timezone.utc)


def _to_utc_ts(df, col="ts"):
    if df is None or df.empty or col not in df:
        return df
    df = df.copy()
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df = df.dropna(subset=[col]).sort_values(col).reset_index(drop=True)
    return df

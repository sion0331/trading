from datetime import datetime, timezone


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

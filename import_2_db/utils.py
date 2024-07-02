from datetime import datetime, timedelta

default_start_date = datetime(2019, 1, 1)


def calculate_day_diff(start_date: datetime, end_date: datetime) -> int:
    return (end_date - start_date).days


def get_next_day(day: datetime) -> datetime:
    one_day = timedelta(days=1)
    next_day = day + one_day
    return datetime.combine(next_day, datetime.min.time())
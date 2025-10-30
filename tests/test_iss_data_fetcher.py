from src.modules.iss_data_fetcher import FetchData
import pytest

def test_fetch_data():
    fetch_data = FetchData()
    longitude, latitude = next(fetch_data)
    assert longitude is not None
    assert latitude is not None

def test_iteration():
    fetch_data = FetchData()
    for _ in range(10):
        longitude, latitude = next(fetch_data)
        assert isinstance(longitude, str) and isinstance(latitude, str)

def test_error_handling():
    fetch_data = FetchData('invalid_url')
    with pytest.raises(StopIteration):
        longitude, latitude = next(fetch_data)
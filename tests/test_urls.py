from deep_research.urls import canonicalize_url


def test_canonicalize_url_removes_tracking_and_fragment() -> None:
    url = "HTTPS://Example.com:443/path/?b=2&utm_source=x&a=1#section"

    assert canonicalize_url(url) == "https://example.com/path?a=1&b=2"

from app.api.routes.view import _build_attachment_headers


def test_attachment_headers_include_ascii_fallback_and_utf8_filename() -> None:
    header = _build_attachment_headers("病例 01.dcm")["Content-Disposition"]

    assert 'filename="01.dcm"' in header
    assert "filename*=UTF-8''%E7%97%85%E4%BE%8B%2001.dcm" in header


def test_attachment_headers_strip_header_injection_characters() -> None:
    header = _build_attachment_headers('bad"\r\nname;.png')["Content-Disposition"]

    assert "\r" not in header
    assert "\n" not in header
    assert 'filename="bad__name_.png"' in header
    assert "filename*=UTF-8''bad%22_name%3B.png" in header

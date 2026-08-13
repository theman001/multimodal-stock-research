import pytest

from src.data_collection.event_text import (
    _html_to_text,
    _parse_document_error,
    _select_exhibit_filenames,
    _strip_us_cover_page_boilerplate,
)


def test_html_to_text_strips_tags():
    html = "<html><body><p>Hello <b>world</b></p></body></html>"
    assert _html_to_text(html) == "Hello world"


def test_html_to_text_removes_script_and_style_content():
    """script/style 태그 '내용물'까지 텍스트로 딸려나오면 안 된다 — 단순
    text_content()만 쓰면 이 문제가 생김(구현 중 실제로 확인)."""
    html = "<html><body><p>Real content</p><script>ignore.me()</script><style>.x{color:red}</style></body></html>"
    text = _html_to_text(html)
    assert text == "Real content"
    assert "ignore" not in text
    assert "color" not in text


def test_html_to_text_normalizes_whitespace():
    html = "<p>Line1</p>\n\n<p>   Line2   </p>"
    assert _html_to_text(html) == "Line1 Line2"


def test_html_to_text_handles_sgml_wrapped_document():
    """SEC EDGAR 응답은 <DOCUMENT><TYPE>8-K...<TEXT><HTML>...로 감싸져 있다
    (실제 응답으로 확인) — lxml이 이런 비표준 래핑도 문제없이 처리해야 한다."""
    sgml = "<DOCUMENT><TYPE>8-K<TEXT><HTML><BODY><P>Item 2.02 Results</P></BODY></HTML></TEXT></DOCUMENT>"
    text = _html_to_text(sgml)
    assert "Item 2.02 Results" in text


def test_parse_document_error_no_file_returns_empty_not_error():
    """status 014(파일 없음)는 실제 데이터에서 발견된 정상 케이스 — 에러가 아니다."""
    xml = b'<?xml version="1.0" encoding="UTF-8"?><result><status>014</status><message>\xed\x8c\x8c\xec\x9d\xbc\xec\x9d\xb4 \xec\xa1\xb4\xec\x9e\xac\xed\x95\x98\xec\xa7\x80 \xec\x95\x8a\xec\x8a\xb5\xeb\x8b\x88\xeb\x8b\xa4.</message></result>'
    assert _parse_document_error(xml) == ""


def test_parse_document_error_real_error_raises():
    xml = b'<?xml version="1.0" encoding="UTF-8"?><result><status>020</status><message>error</message></result>'
    with pytest.raises(RuntimeError):
        _parse_document_error(xml)


def test_strip_us_cover_page_boilerplate_starts_at_item_section():
    """실제 8-K 5건 파일럿에서 커버페이지 문구 자체가 512토큰을 거의 다 써버려
    exhibit 내용까지 못 닿는 걸 확인했다 — Item 섹션부터 잘라야 한다."""
    text = "UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington D.C. lots of boilerplate here Item 2.02 Results of Operations. Apple reported record profits."
    result = _strip_us_cover_page_boilerplate(text)
    assert result.startswith("Item 2.02")


def test_strip_us_cover_page_boilerplate_returns_original_when_no_item_found():
    """'Item'이 안 보이는 비표준 포맷이면 추측해서 자르지 않고 원문 그대로 반환한다."""
    text = "Some unusual filing without the standard cover page format."
    assert _strip_us_cover_page_boilerplate(text) == text


def test_select_exhibit_filenames_excludes_primary_document():
    items = [{"name": "d1d8k.htm"}, {"name": "d1dex991.htm"}]
    result = _select_exhibit_filenames(items, primary_document="d1d8k.htm")
    assert result == ["d1dex991.htm"]


def test_select_exhibit_filenames_excludes_non_text_files():
    """실제 AAPL 8-K index에 이미지(.jpg)가 섞여있었다 — 감성분석에 못 쓰니 제외해야 한다."""
    items = [{"name": "d1d8k.htm"}, {"name": "d1dex991.htm"}, {"name": "g1.jpg"}, {"name": "R1.xml"}]
    result = _select_exhibit_filenames(items, primary_document="d1d8k.htm")
    assert result == ["d1dex991.htm"]


def test_select_exhibit_filenames_excludes_xbrl_viewer_pages():
    """R1.htm, R2.htm... 은 XBRL 재무제표를 표로 렌더링한 자동생성 페이지라
    서술형 텍스트가 아니다 — 실제 AAPL 8-K에서 진짜 보도자료(EX-99.1)를
    밀어내는 걸 확인해서 추가한 필터."""
    items = [{"name": "d1d8k.htm"}, {"name": "d1dex991.htm"}, {"name": "R1.htm"}, {"name": "R23.htm"}]
    result = _select_exhibit_filenames(items, primary_document="d1d8k.htm")
    assert result == ["d1dex991.htm"]


def test_select_exhibit_filenames_excludes_full_submission_txt():
    """'{accession-no}.txt'는 실제 문서가 아니라 SGML 헤더/placeholder뿐인
    전체 제출묶음 파일이다 — 실제 AAPL 8-K에서 이게 512토큰을 다 잡아먹고
    진짜 exhibit 내용이 하나도 안 들어가는 걸 확인해서 추가한 필터."""
    items = [
        {"name": "d1d8k.htm"},
        {"name": "d1dex991.htm"},
        {"name": "0001193125-15-021857.txt"},
    ]
    result = _select_exhibit_filenames(items, primary_document="d1d8k.htm")
    assert result == ["d1dex991.htm"]


def test_select_exhibit_filenames_excludes_index_pages():
    """'{accession-no}-index.html'은 실제 exhibit이 아니라 SEC의 'Filing
    Detail' 목록 페이지(문서 목록/네비게이션 텍스트뿐)다 — .txt만 걸러내면
    될 줄 알았는데 이것도 새는 걸 실제 AAPL 8-K로 뒤늦게 확인해서 추가."""
    items = [
        {"name": "d1d8k.htm"},
        {"name": "d1dex991.htm"},
        {"name": "0001193125-15-149607-index.html"},
        {"name": "0001193125-15-149607-index-headers.html"},
    ]
    result = _select_exhibit_filenames(items, primary_document="d1d8k.htm")
    assert result == ["d1dex991.htm"]


def test_select_exhibit_filenames_caps_at_max_exhibits():
    items = [{"name": "d1d8k.htm"}] + [{"name": f"ex{i}.htm"} for i in range(10)]
    result = _select_exhibit_filenames(items, primary_document="d1d8k.htm")
    assert len(result) == 5

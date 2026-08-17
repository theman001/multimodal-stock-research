"""이벤트(공시/8-K) 원문 텍스트 수집 — 감성 스코어링(모듈 3) 전처리 단계.

모듈 1은 공시 "목록"(메타데이터)만 수집했다 — report_nm/rcept_no(KR),
primaryDocument 파일명(US)뿐이고 실제 본문 텍스트는 없었다(round 3 review
이후 착수 시점에 확인). 감성분석을 하려면 본문이 필요하므로, 이 모듈에서
각 공시/filing의 원문을 받아와 HTML 태그를 제거한 순수 텍스트로 정제해
raw 캐싱한다.

- US: SEC EDGAR Archives에서 8-K의 primaryDocument를 직접 다운로드한다.
  URL 패턴은 `.../edgar/data/{cik}/{accession_no_무대시}/{primaryDocument}`이며
  CIK는 0-padding 없이 그대로 쓴다(실제 요청으로 확인). 내용은 SGML로 감싼
  HTML이라 lxml로 파싱해도 문제없다.

  **주의**: primaryDocument는 대부분 "본문 첨부 참고" 정도의 커버페이지뿐이고,
  실제 실적/보도자료 같은 감성이 실린 내용은 별도 exhibit 파일(EX-99.1 등)에
  있다(파일럿 검증 중 AAPL 8-K로 실제 확인 — primaryDocument만 스코어링하면
  전부 Neutral로 나옴). 그래서 filing index(`index.json`)를 같이 조회해서
  HTML/텍스트 exhibit들도 함께 가져와 본문 뒤에 이어붙인다(이미지/XBRL XML
  등 비-텍스트 첨부는 제외).
- KR: DART `document.xml` API(rcept_no 기반)로 ZIP을 받는다. 내부 문서는
  **EUC-KR로 인코딩**돼 있다(UTF-8 아님 — 실제 응답으로 확인, 안 맞추면
  한글이 깨진다). 일부 rcept_no는 원문 파일 자체가 없다(status 014 "파일이
  존재하지 않습니다" — 정상 케이스, 실제 데이터에서 발견됨) — 이땐 빈 문자열로
  캐싱하고 넘어간다.

두 경우 다 `<script>`/`<style>` 태그를 제거한 뒤 텍스트만 추출하고
공백/개행을 정규화한다.
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import requests
from lxml import html as lxml_html

from src.data_collection.event_mapping import _dart_api_key, _sec_user_agent

SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no}/{filename}"
SEC_ARCHIVES_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no}/index.json"
DART_DOCUMENT_URL = "https://opendart.fss.or.kr/api/document.xml"

# SEC/DART 둘 다 같은 호스트로 문서당 여러 번(US는 최대 7번) 요청이 간다.
# requests.get()을 매번 새로 부르면 요청마다 TCP/TLS 핸드셰이크를 새로 맺어서
# (파일럿 실측 중 일부 filing이 865초까지 걸리는 걸 확인 — 문서 크기(125KB)로는
# 설명이 안 되는 수준) 지연이 누적된다. Session으로 연결을 재사용(keep-alive)
# 하면 이 오버헤드를 줄일 수 있다 — requests.Session은 호스트별로 커넥션 풀을
# 따로 관리하므로 SEC/DART가 같은 Session 객체를 공유해도 문제없다. (최초
# US 쪽만 이 최적화를 적용했다가 KR/DART 쪽은 빠뜨려서 뒤늦게 발견 — KR
# 실행이 눈에 띄게 느린 원인 중 하나였다.)
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session
_REQUEST_DELAY_SECONDS = 0.15
_STATUS_NO_FILE = "014"  # "파일이 존재하지 않습니다" — 에러가 아니라 정상 케이스
_TEXT_EXTENSIONS = (".htm", ".html", ".txt")
_MAX_EXHIBITS = 5  # 첨부가 아주 많은 filing(대형 M&A 계약 등)에서 요청이 무한정 늘어나지 않도록 상한
_XBRL_VIEWER_PATTERN = re.compile(r"^R\d+\.htm$", re.IGNORECASE)  # XBRL 자동생성 뷰어 페이지(R1.htm...) — 서술형 텍스트 아님
# SEC가 시스템적으로 붙이는 "문서가 아닌" 부산물들 — 전부 accession 번호를
# 접두사로 쓰는 규칙적인 이름이라 하나의 패턴으로 묶는다. 실제 요청해보니:
#   - "{accession}.txt"(전체 제출묶음): 실제 문서 대신 SGML 헤더/
#     "Document N - file: xxx.htm" placeholder만 있음
#   - "{accession}-index.html"(Filing Detail 페이지): "EDGAR Filing Documents
#     for... Document Format Files Seq Description..." 같은 목록/네비게이션
#     텍스트만 있음
#   - "{accession}-index-headers.html": 위와 유사한 헤더 정보 페이지
# 이것들을 걸러내지 않으면 진짜 exhibit(EX-99.1 등) 대신 이 쓸모없는 텍스트가
# 512토큰 예산을 다 잡아먹는다(파일럿 검증 중 둘 다 실제로 발견 — 처음엔
# .txt만 걸러내면 될 줄 알았는데 -index.html이 또 새는 걸 뒤늦게 확인함).
_SEC_SYSTEM_FILE_PATTERN = re.compile(r"^\d{10}-\d{2}-\d{6}(\.txt|-index(-headers)?\.html?)$", re.IGNORECASE)


_XML_DECLARATION_PATTERN = re.compile(r"^\s*<\?xml[^>]*\?>")


def _html_to_text(raw_html: str | bytes) -> str:
    """HTML에서 script/style을 제거하고 순수 텍스트만 추출, 공백을 정규화한다.

    DART 문서 중 일부는 원문 자체가 `<?xml version="1.0" encoding="euc-kr"?>`
    로 시작한다. 이미 우리가 EUC-KR로 디코딩해서 파이썬 str로 넘기는데, 그
    str 안에 인코딩 선언이 남아있으면 lxml이 "이미 유니코드인데 인코딩
    선언이 있는 건 모순"이라며 파싱을 거부한다(Colab 전체 실행 중 실제로
    KR 22,490건 중 1,009건이 이 에러로 실패하는 걸 확인:
    `ValueError: Unicode strings with encoding declaration are not
    supported`). 인코딩은 이미 올바르게 적용됐으니 선언 줄만 제거하면 된다
    — 다시 디코딩하는 게 아니라 이미 디코딩된 텍스트에서 모순되는 선언
    문구만 지우는 것.
    """
    if isinstance(raw_html, str):
        raw_html = _XML_DECLARATION_PATTERN.sub("", raw_html, count=1)
    doc = lxml_html.fromstring(raw_html)
    for bad in doc.xpath("//script | //style"):
        bad.getparent().remove(bad)
    return " ".join(doc.text_content().split())


_ITEM_SECTION_PATTERN = re.compile(r"\bItem\s+\d")


def _strip_us_cover_page_boilerplate(text: str) -> str:
    """8-K 커버페이지의 정형 문구("UNITED STATES SECURITIES AND EXCHANGE
    COMMISSION... Check the appropriate box...")를 잘라내고 실제 내용이
    시작되는 "Item N" 지점부터 반환한다.

    FinBERT는 512 토큰까지만 보는데, 이 커버페이지 문구가 모든 8-K에서
    거의 동일하게 ~250단어(약 500토큰)를 차지해서 실제 exhibit 내용까지
    닿기도 전에 토큰 한도를 다 써버린다(파일럿 검증 중 AAPL 8-K 5건 전부에서
    실제로 확인 — 커버페이지만으로 스코어링 결과가 전부 Neutral로 나왔음).
    "Item "이 안 보이면(비표준 포맷) 원문을 그대로 반환한다 — 추측해서 자르지
    않는다.
    """
    match = _ITEM_SECTION_PATTERN.search(text)
    return text[match.start() :] if match else text


def _select_exhibit_filenames(index_items: list[dict], primary_document: str) -> list[str]:
    """filing index에서 primaryDocument 외의 텍스트성 첨부(exhibit) 파일명만 골라낸다.

    이미지(.jpg/.gif)나 XBRL 구조화 데이터(.xml/.xsd) 등은 감성분석에 쓸 수
    없는 첨부라 제외한다. XBRL이 첨부된 8-K는 "R1.htm, R2.htm..." 같은
    자동생성 뷰어 페이지(재무제표를 표로 렌더링한 것)가 수십 개씩 딸려오는데,
    이것도 서술형 텍스트가 아니라 감성분석에 쓸모없다 — 실제 AAPL 8-K로 확인:
    걸러내지 않으면 이 페이지들이 진짜 보도자료(EX-99.1) 자리를 밀어냄.
    "{accession-no}.txt"/"-index.html" 같은 SEC 시스템 부산물(실제 문서 대신
    placeholder/네비게이션 텍스트만 있음)도 같은 이유로 제외한다. 첨부가 너무
    많은 filing(대형 계약서 등)에서 요청이 무한정 늘어나지 않도록
    `_MAX_EXHIBITS`로 상한을 둔다.
    """
    names = [
        item["name"]
        for item in index_items
        if item["name"] != primary_document
        and item["name"].lower().endswith(_TEXT_EXTENSIONS)
        and not _XBRL_VIEWER_PATTERN.match(item["name"])
        and not _SEC_SYSTEM_FILE_PATTERN.match(item["name"])
    ]
    return names[:_MAX_EXHIBITS]


def fetch_us_filing_text(
    cik: int, accession_number: str, primary_document: str, raw_dir: Path, force: bool = False
) -> str:
    """8-K primaryDocument + 텍스트성 exhibit들을 다운로드해 정제된 텍스트로
    raw 캐싱한다 (캐시 우선). primaryDocument는 대개 커버페이지뿐이고 실제
    보도자료 등 내용은 exhibit(EX-99.1 등)에 있는 경우가 많아 같이 가져온다.
    """
    cache_dir = raw_dir / "events" / "text"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"us_{accession_number}.txt"
    if cache_path.exists() and not force:
        return cache_path.read_text(encoding="utf-8")

    headers = {"User-Agent": _sec_user_agent()}
    accession_no = accession_number.replace("-", "")
    session = _get_session()

    def _fetch_one(filename: str) -> str:
        url = SEC_ARCHIVES_URL.format(cik=cik, accession_no=accession_no, filename=filename)
        resp = session.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        time.sleep(_REQUEST_DELAY_SECONDS)
        return _html_to_text(resp.content)

    parts = [_strip_us_cover_page_boilerplate(_fetch_one(primary_document))]

    index_url = SEC_ARCHIVES_INDEX_URL.format(cik=cik, accession_no=accession_no)
    index_resp = session.get(index_url, headers=headers, timeout=30)
    index_resp.raise_for_status()
    time.sleep(_REQUEST_DELAY_SECONDS)
    index_items = index_resp.json()["directory"]["item"]

    for filename in _select_exhibit_filenames(index_items, primary_document):
        parts.append(_fetch_one(filename))

    text = " ".join(p for p in parts if p)
    cache_path.write_text(text, encoding="utf-8")
    return text


def _parse_document_error(xml_bytes: bytes) -> str:
    """document.xml이 zip이 아니라 에러 XML을 준 경우를 처리한다.

    status 014(원문 파일 없음)는 에러가 아니라 정상 케이스라 빈 문자열을
    반환한다. 그 외 상태 코드(인증키 오류, 트래픽 초과 등)는 조용히 넘기지
    않고 raise한다 — events_kr.py의 _STATUS_NO_DATA와 동일한 원칙.
    """
    payload = ElementTree.fromstring(xml_bytes)
    status = payload.findtext("status")
    if status == _STATUS_NO_FILE:
        return ""
    raise RuntimeError(f"DART document.xml 오류 (status={status}): {payload.findtext('message')}")


def fetch_kr_disclosure_text(rcept_no: str, raw_dir: Path, force: bool = False) -> str:
    """DART document.xml로 공시 원문을 받아 정제된 텍스트로 raw 캐싱한다 (캐시 우선).

    원문 파일이 없는 rcept_no(status 014)는 빈 문자열로 캐싱한다.
    """
    cache_dir = raw_dir / "events" / "text"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"kr_{rcept_no}.txt"
    if cache_path.exists() and not force:
        return cache_path.read_text(encoding="utf-8")

    api_key = _dart_api_key()
    session = _get_session()
    resp = session.get(DART_DOCUMENT_URL, params={"crtfc_key": api_key, "rcept_no": rcept_no}, timeout=30)
    resp.raise_for_status()
    time.sleep(_REQUEST_DELAY_SECONDS)

    raw_content = resp.content
    if not zipfile.is_zipfile(io.BytesIO(raw_content)):
        text = _parse_document_error(raw_content)
    else:
        with zipfile.ZipFile(io.BytesIO(raw_content)) as zf:
            raw_bytes = zf.read(zf.namelist()[0])
        raw_html = raw_bytes.decode("euc-kr", errors="replace")
        text = _html_to_text(raw_html)

    cache_path.write_text(text, encoding="utf-8")
    return text

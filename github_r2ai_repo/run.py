import argparse
import base64
import html as html_lib
import json
import re
import time
import warnings
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from unidecode import unidecode


warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style, apply openpyxl's default",
    category=UserWarning,
)

BASE_OLD = "https://vbpl.vn/TW/Pages"
BASE_NEW = "https://vbpl.vn/van-ban/chi-tiet"
GATEWAY_API = "https://vbpl-bientap-gateway.moj.gov.vn/api"
DETAIL_ACTION = "0fb12b3561faa05adec51a82efb3e4f4f427f07b"
DIAGRAM_ACTION = "4a3423ce75290ef83a022333ee187acf4d38d3fb"
HISTORY_ACTION = "45bee55da28429892bec658192db2b75a74b6256"
RELATED_FILE_ACTION = "ae8ddf86413544ad267f753f6b61a5f71dd260b2"
FILE_DOWNLOAD_ACTION = "11972afaf856c3e135f5835fb529ce24ba23bee2"

HEADERS = {"User-Agent": "Mozilla/5.0 R2AI-VBPL-Crawler/1.0"}
TIMEOUT = 30
SLEEP = 0.8
REQUEST_RETRIES = 3
RETRY_BACKOFF = 1.5


def request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("timeout", TIMEOUT)
    last_error = None
    last_response = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code < 500:
                return response
            last_response = response
            last_error = RuntimeError(f"HTTP {response.status_code}")
        except requests.RequestException as exc:
            last_error = exc
        if attempt < REQUEST_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)
    if last_response is not None:
        return last_response
    raise last_error


def request_get(url: str, **kwargs) -> requests.Response:
    return request_with_retry("GET", url, **kwargs)


def request_post(url: str, **kwargs) -> requests.Response:
    return request_with_retry("POST", url, **kwargs)


def fix_mojibake_text(value: str) -> str:
    if not isinstance(value, str):
        return value
    markers = ("Ã", "Ä", "áº", "á»", "Æ", "Â", "â€")
    if not any(marker in value for marker in markers):
        return value
    try:
        fixed = value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    bad_before = sum(value.count(marker) for marker in markers)
    bad_after = sum(fixed.count(marker) for marker in markers)
    return fixed if bad_after < bad_before else value


def fix_mojibake_data(value):
    if isinstance(value, str):
        return fix_mojibake_text(value)
    if isinstance(value, list):
        return [fix_mojibake_data(item) for item in value]
    if isinstance(value, dict):
        return {key: fix_mojibake_data(item) for key, item in value.items()}
    return value


def read_csv_auto(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if path_obj.is_dir():
        files = sorted(
            [
                *path_obj.glob("*.csv"),
                *path_obj.glob("*.xlsx"),
                *path_obj.glob("*.xls"),
            ]
        )
        files = [file for file in files if not file.name.startswith("~$")]
        if not files:
            raise RuntimeError(f"Không tìm thấy file CSV/XLSX trong thư mục: {path}")
        frames = [read_csv_auto(str(file)) for file in files]
        combined = pd.concat(frames, ignore_index=True)
        id_col = next((col for col in combined.columns if str(col).strip().lower() == "id"), None)
        if id_col:
            combined = combined.drop_duplicates(subset=[id_col], keep="first")
        return combined

    with open(path, "rb") as f:
        if f.read(2) == b"PK":
            return pd.read_excel(path)

    encodings = ["utf-8-sig", "utf-8", "cp1258", "utf-16"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Không đọc được CSV. Lỗi cuối: {last_err}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        c_norm = str(c).strip().lower()
        c_ascii = unidecode(c_norm)
        if c_norm in ["id", "doc_id", "uuid", "itemid", "item_id"]:
            col_map[c] = "id"
        elif "so ky hieu" in c_ascii or "so hieu" in c_ascii:
            col_map[c] = "so_ky_hieu"
        elif "ten van ban" in c_ascii or "trich yeu" in c_ascii:
            col_map[c] = "ten_van_ban"
        elif "loai van ban" in c_ascii:
            col_map[c] = "loai_van_ban"
        elif "url" in c_norm:
            col_map[c] = "url"

    df = df.rename(columns=col_map)
    if "id" not in df.columns:
        raise ValueError("CSV cần có cột ID/id/item_id/uuid.")

    for c in ["so_ky_hieu", "ten_van_ban", "loai_van_ban", "url"]:
        if c not in df.columns:
            df[c] = ""
    df["id"] = df["id"].astype(str).str.strip()
    return df


def slugify_vi(text: str, max_len: int = 120) -> str:
    text = unidecode(str(text or "").strip().lower())
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len].strip("-") or "van-ban"


def is_numeric_item_id(doc_id: str) -> bool:
    return str(doc_id).strip().isdigit()


def candidate_urls(row: dict) -> list[str]:
    doc_id = str(row.get("id", "")).strip()
    title = str(row.get("ten_van_ban", "") or row.get("so_ky_hieu", "")).strip()
    url = str(row.get("url", "")).strip()
    urls = []

    if is_numeric_item_id(doc_id):
        urls.extend([
            f"{BASE_OLD}/vbpq-toanvan.aspx?ItemID={doc_id}",
            f"{BASE_OLD}/vbpq-print.aspx?ItemID={doc_id}",
            f"{BASE_NEW}/--{doc_id}",
        ])
    else:
        if url and url.startswith("http"):
            urls.append(url)
        slug = slugify_vi(title)
        urls.extend([f"{BASE_NEW}/{slug}--{doc_id}", f"{BASE_NEW}/--{doc_id}"])

    if url and url.startswith("http"):
        urls.append(url)

    out, seen = [], set()
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def fetch(url: str) -> tuple[str | None, str | None]:
    try:
        r = request_get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        time.sleep(SLEEP)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        r.encoding = "utf-8"
        html = fix_mojibake_text(r.text)
        low = html[:5000].lower()
        bad_markers = ["không tìm thấy", "not found", "404", "internal server error"]
        if any(m in low for m in bad_markers):
            return None, "Page looks invalid"
        return html, None
    except Exception as e:
        return None, str(e)


def fetch_first_ok(urls: list[str]) -> tuple[str | None, str | None, list[dict]]:
    logs = []
    for url in urls:
        html, err = fetch(url)
        logs.append({"url": url, "ok": html is not None, "error": err})
        if html:
            return html, url, logs
    return None, None, logs


def extract_next_action_html(text: str) -> str | None:
    match = re.search(r"\d+:T[0-9a-fA-F]+,(<html>.*?</html>)", text, re.S)
    return match.group(1) if match else None


def fetch_gateway_document_content_html(doc_id: str) -> tuple[str | None, str | None]:
    try:
        r = request_get(f"{GATEWAY_API}/qtdc/public/doc/{doc_id}", headers=HEADERS, timeout=TIMEOUT)
        time.sleep(SLEEP)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        r.encoding = "utf-8"
        payload = r.json()
        payload = fix_mojibake_data(payload)
        html = (((payload.get("data") or {}).get("documentContent") or {}).get("content"))
        if not html:
            return None, "No documentContent.content in gateway response"
        return html, None
    except Exception as e:
        return None, str(e)


def fetch_document_content_html(doc_id: str, page_url: str | None) -> tuple[str | None, str | None]:
    if not doc_id:
        return None, None

    gateway_html, gateway_err = fetch_gateway_document_content_html(doc_id)
    if gateway_html:
        return gateway_html, None

    url = page_url or f"{BASE_NEW}/--{doc_id}"
    headers = {
        **HEADERS,
        "Next-Action": DETAIL_ACTION,
        "Content-Type": "text/plain;charset=UTF-8",
        "Accept": "text/x-component",
    }
    try:
        r = request_post(url, headers=headers, data=json.dumps([doc_id]), timeout=TIMEOUT)
        time.sleep(SLEEP)
        r.encoding = "utf-8"
        if r.status_code != 200:
            return None, f"Gateway: {gateway_err}; Action HTTP {r.status_code}"
        html = fix_mojibake_text(extract_next_action_html(r.text) or "")
        if not html:
            return None, f"Gateway: {gateway_err}; Action: No content HTML in response"
        return html, None
    except Exception as e:
        return None, f"Gateway: {gateway_err}; Action: {e}"


def extract_next_action_json(text: str):
    match = re.search(r"^1:(\[.*\]|\{.*\})$", text, re.M)
    if not match:
        match = re.search(r"^(?!0:)\d+:(\[.*\]|\{.*\})$", text, re.M)
    return fix_mojibake_data(json.loads(match.group(1))) if match else None


def fetch_next_action_json(doc_id: str, page_url: str | None, action_id: str, payload=None) -> tuple[dict | list | None, str | None]:
    if not doc_id:
        return None, "Empty document id"
    url = page_url or f"{BASE_NEW}/--{doc_id}"
    headers = {
        **HEADERS,
        "Next-Action": action_id,
        "Content-Type": "text/plain;charset=UTF-8",
        "Accept": "text/x-component",
    }
    try:
        body = payload if payload is not None else [doc_id]
        r = request_post(url, headers=headers, data=json.dumps(body, ensure_ascii=False), timeout=TIMEOUT)
        time.sleep(SLEEP)
        r.encoding = "utf-8"
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        data = extract_next_action_json(r.text)
        if data is None:
            return None, "No JSON in action response"
        return data, None
    except Exception as e:
        return None, str(e)


def extract_next_action_blob(text: str) -> tuple[bytes | None, str | None]:
    payload_match = re.search(r"^(\d+):T\d+,(.*)$", text, re.M)
    meta_match = re.search(r"^\d+:(\{.*\})$", text, re.M)
    if not payload_match or not meta_match:
        return None, "No blob payload in action response"
    payload_id = payload_match.group(1)
    meta = json.loads(meta_match.group(1))
    if meta.get("data") != f"${payload_id}":
        return None, "Blob metadata does not reference payload"
    return base64.b64decode(payload_match.group(2)), None


def fetch_related_files(doc_id: str, page_url: str | None) -> tuple[list[dict], str | None]:
    if not doc_id or is_numeric_item_id(doc_id):
        return [], None
    files, err = fetch_next_action_json(doc_id, page_url, RELATED_FILE_ACTION)
    if not isinstance(files, list):
        return [], err or "No related file list in action response"
    return files, None


RELATION_LABELS = {
    "1": "Bãi bỏ / Thay thế",
    "2": "Được hướng dẫn áp dụng",
    "3": "Căn cứ ban hành",
    "4": "Được quy định chi tiết",
    "5": "Được hợp nhất",
    "6": "Được đính chính",
    "7": "Bị đình chỉ thi hành",
    "8": "Bị tạm ngưng hiệu lực",
    "9": "Được công bố",
    "10": "Sửa đổi, bổ sung",
    "11": "Bãi bỏ",
    "12": "Thay thế",
    "13": "Dẫn chiếu",
    "14": "Áp dụng",
}


def normalize_relation_items(groups: dict | None) -> list[dict]:
    out = []
    if not isinstance(groups, dict):
        return out
    for rel_id, items in groups.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            target_id = str(item.get("id") or "").strip()
            out.append({
                "quan_he_id": str(rel_id),
                "quan_he": RELATION_LABELS.get(str(rel_id), str(rel_id)),
                "id": target_id or None,
                "name": item.get("name"),
                "url": f"{BASE_NEW}/--{target_id}" if target_id else None,
            })
    return out


def fetch_diagram(doc_id: str, page_url: str | None) -> tuple[dict, str | None]:
    data, err = fetch_next_action_json(doc_id, page_url, DIAGRAM_ACTION)
    if not isinstance(data, dict):
        return {"van_ban_duoc_tac_dong": [], "van_ban_tac_dong": []}, err
    return {
        "van_ban_duoc_tac_dong": normalize_relation_items(data.get("documentNamesByType")),
        "van_ban_tac_dong": normalize_relation_items(data.get("documentNamesBySource")),
    }, None


def fetch_history(doc_id: str, page_url: str | None) -> tuple[list[dict], str | None]:
    data, err = fetch_next_action_json(doc_id, page_url, HISTORY_ACTION)
    if not isinstance(data, dict):
        return [], err
    history = []
    for item in data.get("history") or []:
        if not isinstance(item, dict):
            continue
        history.append({
            "ngay_thay_doi": item.get("createdDate"),
            "loai_thay_doi": item.get("editType"),
            "noi_dung": item.get("content"),
            "nguoi_thay_doi": item.get("createdBy"),
            "van_ban_nguon": item.get("sourceDocumentName"),
        })
    return history, None


def soup_text_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    lines = []
    for line in soup.get_text("\n").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


def tag_text(tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()


def tag_classes(tag) -> set[str]:
    return set(tag.get("class") or [])


def split_articles_from_html(html: str | None) -> list[dict]:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    selected_level = select_html_chunk_level(soup)
    if not selected_level:
        return []

    start_class = {
        "article": "prov-article",
        "section": "prov-section",
        "clause": "prov-clause",
        "part": "prov-part",
    }[selected_level]

    articles = []
    current_part = None
    current_chapter = None
    current_section = None
    current_article = None

    for tag in soup.find_all(["p", "div", "li"]):
        text = tag_text(tag)
        if not text:
            continue
        classes = tag_classes(tag)
        if "prov-part" in classes:
            current_part = text
        if "prov-chapter" in classes:
            current_chapter = text
        if "prov-section" in classes:
            current_section = text

        if start_class in classes:
            if current_article:
                articles.append(current_article)
            chunk_no, chunk_title = parse_structured_heading(text, selected_level, len(articles) + 1)
            current_article = {
                "chunk_level": selected_level,
                "chunk_no": chunk_no,
                "chunk_title": chunk_title,
                "part": current_part,
                "chapter": current_chapter,
                "section": current_section if selected_level != "section" else None,
                "article_no": chunk_no,
                "article_title": chunk_title,
                "provision_id": tag.get("id"),
                "lines": [text],
            }
            continue

        if current_article:
            if selected_level in {"section", "clause"} and "prov-part" in classes:
                articles.append(current_article)
                current_article = None
                continue
            if selected_level == "clause" and "prov-section" in classes:
                articles.append(current_article)
                current_article = None
                continue
            current_article["lines"].append(text)
            continue

        if "prov-chapter" in classes:
            current_chapter = text
            continue
        if "prov-section" in classes:
            current_section = text
            continue
        if "prov-article" in classes:
            match = RE_ARTICLE.match(text)
            article_no = match.group(1) if match else str(len(articles) + 1)
            article_title = match.group(2).strip() if match else text
            if current_article:
                articles.append(current_article)
            current_article = {
                "chapter": current_chapter,
                "section": current_section,
                "article_no": article_no,
                "article_title": article_title,
                "provision_id": tag.get("id"),
                "lines": [text],
            }
            continue
        if current_article:
            current_article["lines"].append(text)

    if current_article:
        articles.append(current_article)

    for art in articles:
        finalize_article(art)
    return articles


def select_html_chunk_level(soup: BeautifulSoup) -> str | None:
    if soup.select(".prov-article"):
        return "article"
    if soup.select(".prov-section"):
        return "section"
    if soup.select(".prov-clause"):
        return "clause"
    if soup.select(".prov-part"):
        return "part"
    return None


def parse_structured_heading(text: str, level: str, index: int) -> tuple[str, str]:
    patterns = {
        "article": RE_ARTICLE,
        "section": re.compile(r"^(\d+(?:[.\-]\d+)*)[\-.]?\s*(.*)", re.I),
        "clause": re.compile(r"^(\d+(?:[.\-]\d+)*)[\-.]?\s*(.*)", re.I),
        "part": re.compile(r"^([IVXLCDM]+|\d+)[\-.]?\s*(.*)", re.I),
    }
    match = patterns[level].match(text)
    if match:
        return match.group(1), match.group(2).strip()
    return f"{level}_{index}", text


RE_CHAPTER = re.compile(r"^Chương\s+([IVXLCDM]+|\d+)\b", re.I)
RE_SECTION = re.compile(r"^Mục\s+\d+\b", re.I)
RE_ARTICLE = re.compile(r"^Điều\s+(\d+[a-zA-Z]?)\.?\s*(.*)", re.I)
RE_CLAUSE = re.compile(r"^(\d+)\.\s+")
RE_POINT = re.compile(r"^([a-zA-ZđĐ])\)\s+")
RE_DOC_NUMBER = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ\-]+(?:-[A-ZĐ]+)*\b", re.I)


def finalize_article(art: dict) -> None:
    clauses = []
    current_clause = None
    for line in art["lines"][1:]:
        cm = RE_CLAUSE.match(line)
        if cm:
            if current_clause:
                clauses.append(current_clause)
            current_clause = {"clause_no": cm.group(1), "text": line, "points": []}
            continue
        pm = RE_POINT.match(line)
        if pm and current_clause:
            current_clause["points"].append({"point": pm.group(1).lower(), "text": line})
            continue
        if current_clause:
            current_clause["text"] += "\n" + line
    if current_clause:
        clauses.append(current_clause)
    art["article_text"] = "\n".join(art["lines"])
    art["clauses"] = clauses
    del art["lines"]


def split_articles(lines: list[str]) -> list[dict]:
    articles = []
    current_chapter = None
    current_section = None
    current_article = None

    for line in lines:
        if RE_CHAPTER.match(line):
            current_chapter = line
            continue
        if RE_SECTION.match(line):
            current_section = line
            continue
        m = RE_ARTICLE.match(line)
        if m:
            if current_article:
                articles.append(current_article)
            current_article = {
                "chunk_level": "article",
                "chunk_no": m.group(1),
                "chunk_title": m.group(2).strip(),
                "chapter": current_chapter,
                "section": current_section,
                "article_no": m.group(1),
                "article_title": m.group(2).strip(),
                "lines": [line],
            }
            continue
        if current_article:
            current_article["lines"].append(line)

    if current_article:
        articles.append(current_article)

    if not articles and lines:
        return [{
            "chunk_level": "full_text",
            "chunk_no": "toan_van",
            "chunk_title": "Toàn văn",
            "chapter": None,
            "section": None,
            "article_no": "toan_van",
            "article_title": "Toàn văn",
            "article_text": "\n".join(lines),
            "clauses": [],
        }]

    for art in articles:
        finalize_article(art)
    return articles


def article_ids_from_html(html: str | None) -> dict[str, str]:
    if not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    out = {}
    for tag in soup.select(".prov-article[id]"):
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True))
        match = RE_ARTICLE.match(text)
        if match:
            out.setdefault(match.group(1), tag.get("id"))
    return out


def extract_doc_numbers(text: str | None) -> set[str]:
    if not text:
        return set()
    return {unidecode(m.group(0)).upper() for m in RE_DOC_NUMBER.finditer(text)}


def relation_doc_number(item: dict) -> str | None:
    name = item.get("name") or ""
    numbers = extract_doc_numbers(name)
    return next(iter(numbers), None) if numbers else None


def enrich_articles_with_relations(articles: list[dict], diagram: dict, content_html: str | None) -> None:
    ids_by_article_no = article_ids_from_html(content_html)
    relation_items = []
    for group_name in ["van_ban_duoc_tac_dong", "van_ban_tac_dong"]:
        for item in diagram.get(group_name) or []:
            if not isinstance(item, dict):
                continue
            doc_number = relation_doc_number(item)
            if doc_number:
                relation_items.append({**item, "relation_group": group_name, "matched_so_hieu": doc_number})

    for article in articles:
        article["provision_id"] = article.get("provision_id") or ids_by_article_no.get(str(article.get("article_no")))
        article_numbers = extract_doc_numbers(article.get("article_text"))
        article["impacted_documents"] = [
            item for item in relation_items
            if item.get("matched_so_hieu") in article_numbers
        ]


def extract_file_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url or "", a["href"])
        text = a.get_text(" ", strip=True).lower()
        href_low = href.lower()
        is_file = any(ext in href_low for ext in [".pdf", ".doc", ".docx"])
        is_download_text = any(k in text for k in ["bản pdf", "tải về", ".pdf", ".doc", ".docx"])
        if is_file or is_download_text:
            links.append(href)

    out, seen = [], set()
    for x in links:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", str(name), flags=re.UNICODE)
    return name[:180]


def related_file_download_url(doc_id: str, file_name: str) -> str:
    quoted = quote(file_name, safe="")
    return f"{GATEWAY_API}/qtdc/public/doc/minio/buckets/vbpl/{doc_id}/{quoted}/download"


def download_file(url: str, out_dir: Path, prefix: str) -> str | None:
    try:
        r = request_get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        time.sleep(SLEEP)
        if r.status_code != 200:
            return None
        content_type = r.headers.get("Content-Type", "").lower()
        suffix = Path(r.url.split("?")[0]).suffix.lower()
        if not suffix:
            suffix = ".pdf" if "pdf" in content_type else ".doc" if "word" in content_type or "msword" in content_type else ".bin"
        path = out_dir / safe_filename(prefix + suffix)
        path.write_bytes(r.content)
        return str(path)
    except Exception:
        return None


def download_related_file(doc_id: str, file_name: str, out_dir: Path, prefix: str) -> tuple[str | None, str | None]:
    if not file_name:
        return None, "Empty file name"
    try:
        headers = {
            **HEADERS,
            "Next-Action": FILE_DOWNLOAD_ACTION,
            "Content-Type": "text/plain;charset=UTF-8",
            "Accept": "text/x-component",
        }
        body = [{"bucketName": "vbpl", "folderName": doc_id, "objectName": file_name}]
        r = request_post(
            "https://vbpl.vn/van-ban/dia-phuong",
            headers=headers,
            data=json.dumps(body, ensure_ascii=False),
            timeout=TIMEOUT,
        )
        time.sleep(SLEEP)
        action_err = None
        if r.status_code == 200:
            content, err = extract_next_action_blob(r.text)
            if content:
                path = out_dir / safe_filename(f"{prefix}_{file_name}")
                path.write_bytes(content)
                return str(path), None
            action_err = err
        else:
            action_err = f"Action HTTP {r.status_code}: {r.text[:200]}"

        r = request_get(related_file_download_url(doc_id, file_name), headers=HEADERS, timeout=TIMEOUT)
        time.sleep(SLEEP)
        if r.status_code != 200:
            return None, f"{action_err}; Direct HTTP {r.status_code}: {r.text[:200]}"
        path = out_dir / safe_filename(f"{prefix}_{file_name}")
        if not path.suffix:
            path = path.with_suffix(Path(file_name).suffix.lower() or ".bin")
        path.write_bytes(r.content)
        return str(path), None
    except Exception as e:
        return None, str(e)


def docx_text_lines(path: str) -> list[str]:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    lines = []
    for paragraph in root.findall(".//w:p", ns):
        parts = [node.text for node in paragraph.findall(".//w:t", ns) if node.text]
        text = html_lib.unescape("".join(parts))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            lines.append(text)
    return lines


def crawl_one(row: dict, out_dir: Path) -> dict:
    doc_id = str(row["id"]).strip()
    html, final_url, logs = fetch_first_ok(candidate_urls(row))
    result = {
        "id": doc_id,
        "so_ky_hieu": row.get("so_ky_hieu", ""),
        "ten_van_ban": row.get("ten_van_ban", ""),
        "loai_van_ban": row.get("loai_van_ban", ""),
        "source_url": final_url,
        "candidate_logs": logs,
        "ok": html is not None,
        "raw_html_path": None,
        "content_html_path": None,
        "content_text_path": None,
        "content_fetch_error": None,
        "diagram_fetch_error": None,
        "history_fetch_error": None,
        "related_file_error": None,
        "related_file_download_errors": [],
        "file_paths": [],
        "luoc_do": {"van_ban_duoc_tac_dong": [], "van_ban_tac_dong": []},
        "lich_su": [],
        "article_count": 0,
        "articles": [],
    }
    if not html:
        return result

    raw_dir = out_dir / "raw_html"
    file_dir = out_dir / "files"
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{safe_filename(doc_id)}.html"
    raw_path.write_text(html, encoding="utf-8")
    result["raw_html_path"] = str(raw_path)

    content_html, content_err = fetch_document_content_html(doc_id, final_url)
    result["content_fetch_error"] = content_err

    diagram, diagram_err = fetch_diagram(doc_id, final_url)
    history, history_err = fetch_history(doc_id, final_url)
    result["luoc_do"] = diagram
    result["lich_su"] = history
    result["diagram_fetch_error"] = diagram_err
    result["history_fetch_error"] = history_err

    parse_html = content_html or html
    if content_html:
        content_path = raw_dir / f"{safe_filename(doc_id)}.content.html"
        content_path.write_text(content_html, encoding="utf-8")
        result["content_html_path"] = str(content_path)

    lines = soup_text_lines(parse_html)
    articles = split_articles_from_html(parse_html)
    if not articles:
        articles = split_articles(lines)

    if not articles:
        related_files, related_err = fetch_related_files(doc_id, final_url)
        result["related_file_error"] = related_err
        for i, item in enumerate(related_files, start=1):
            file_name = str(item.get("fileName") or "").strip()
            fpath, ferr = download_related_file(doc_id, file_name, file_dir, f"{doc_id}_{i}")
            if fpath:
                result["file_paths"].append(fpath)
            if ferr:
                result["related_file_download_errors"].append({"fileName": file_name, "error": ferr})

        docx_path = next((p for p in result["file_paths"] if p.lower().endswith(".docx")), None)
        if docx_path:
            lines = docx_text_lines(docx_path)
            articles = split_articles(lines)
            text_path = raw_dir / f"{safe_filename(doc_id)}.content.txt"
            text_path.write_text("\n".join(lines), encoding="utf-8")
            result["content_text_path"] = str(text_path)

    result["article_count"] = len(articles)
    enrich_articles_with_relations(articles, result["luoc_do"], content_html)
    result["articles"] = articles

    for i, link in enumerate(extract_file_links(html, final_url), start=1):
        fpath = download_file(link, file_dir, f"{doc_id}_{i}")
        if fpath:
            result["file_paths"].append(fpath)
    return result


def read_text_if_exists(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else None


def build_r2ai_document(result: dict) -> dict:
    content_html = read_text_if_exists(result.get("content_html_path"))
    if content_html is None:
        text = read_text_if_exists(result.get("content_text_path"))
        content_html = f"<html><body><pre>{html_lib.escape(text)}</pre></body></html>" if text else None

    content_text = "\n".join(soup_text_lines(content_html or ""))
    files = []
    for i, fpath in enumerate(result.get("file_paths") or [], start=1):
        p = Path(fpath)
        ext = p.suffix.lower().lstrip(".")
        files.append({
            "file_name": p.name,
            "loai_file": ext.upper() if ext else None,
            "file_title": p.name,
            "file_order": i,
            "url": str(p),
        })

    return {
        "doc_id": result["id"],
        "thuoc_tinh": {
            "id": result["id"],
            "so_hieu": result.get("so_ky_hieu") or None,
            "ten_van_ban": result.get("ten_van_ban") or None,
            "loai_van_ban": result.get("loai_van_ban") or None,
            "loai_code": None,
            "ngay_ban_hanh": None,
            "hieu_luc_tu": None,
            "hieu_luc_den": None,
            "trang_thai_hieu_luc": None,
            "trang_thai_code": None,
            "co_quan_ban_hanh": None,
            "bo_nganh": None,
            "ngon_ngu": "vi",
            "luot_xem": None,
            "ngay_cap_nhat": None,
            "trang_thai_xuat_ban": None,
            "is_luat": None,
            "is_cu": None,
            "has_content": bool(content_html),
            "file_pdf": next((Path(p).name for p in result.get("file_paths") or [] if p.lower().endswith(".pdf")), None),
            "file_doc": next((Path(p).name for p in result.get("file_paths") or [] if p.lower().endswith((".doc", ".docx"))), None),
            "url_chi_tiet": result.get("source_url"),
            "url_pdf": None,
            "url_doc": None,
        },
        "noi_dung": {
            "id": result["id"],
            "title": result.get("ten_van_ban") or None,
            "content_html": content_html,
            "content_length": len(content_text),
        },
        "luoc_do": result.get("luoc_do") or {"van_ban_duoc_tac_dong": [], "van_ban_tac_dong": []},
        "lich_su": result.get("lich_su") or [],
        "files_dinh_kem": files,
        "crawl_errors": {
            "content_fetch_error": result.get("content_fetch_error"),
            "diagram_fetch_error": result.get("diagram_fetch_error"),
            "history_fetch_error": result.get("history_fetch_error"),
            "related_file_error": result.get("related_file_error"),
            "related_file_download_errors": result.get("related_file_download_errors") or [],
        },
        "crawled_at": datetime.now().isoformat(timespec="seconds"),
    }


def build_r2ai_article_chunk(result: dict, article: dict) -> str:
    level = str(article.get("chunk_level") or "article").strip()
    number = str(article.get("chunk_no") or article.get("article_no") or "").strip()
    title = str(article.get("chunk_title") or article.get("article_title") or "").strip()
    label = " ".join(p for p in [level, number, title] if p)
    parts = [
        str(result.get("so_ky_hieu") or "").strip(),
        str(result.get("ten_van_ban") or "").strip(),
        label,
        str(article.get("article_text") or "").strip(),
    ]
    return "\n".join(p for p in parts if p)


def unique_article_key(chunks: dict, doc_id: str, article_no: str) -> str:
    base = f"{doc_id}_{safe_filename(str(article_no))}"
    key = base
    i = 2
    while key in chunks:
        key = f"{base}_{i}"
        i += 1
    return key


def main():
    global SLEEP, TIMEOUT, REQUEST_RETRIES, RETRY_BACKOFF

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="van-ban-da-cong-khai.csv", help="Đường dẫn file CSV/XLSX đầu vào")
    parser.add_argument("--out", default="vbpl_crawled", help="Thư mục output")
    parser.add_argument("--start", type=int, default=0, help="Dòng bắt đầu, tính từ 0")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số dòng để test")
    parser.add_argument("--workers", type=int, default=1, help="Số luồng crawl song song")
    parser.add_argument("--sleep", type=float, default=SLEEP, help="Thời gian nghỉ sau mỗi request")
    parser.add_argument("--timeout", type=int, default=TIMEOUT, help="Timeout mỗi request, tính bằng giây")
    parser.add_argument("--retries", type=int, default=REQUEST_RETRIES, help="Số lần thử lại khi request lỗi")
    parser.add_argument("--backoff", type=float, default=RETRY_BACKOFF, help="Thời gian backoff giữa các lần retry")
    parser.add_argument("--no-r2ai", action="store_true", help="Không xuất cấu trúc JSON giống R2AI_VBPL")
    args = parser.parse_args()

    SLEEP = max(args.sleep, 0)
    TIMEOUT = max(args.timeout, 1)
    REQUEST_RETRIES = max(args.retries, 1)
    RETRY_BACKOFF = max(args.backoff, 0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_columns(read_csv_auto(args.csv))
    if args.start:
        df = df.iloc[args.start:]
    if args.limit:
        df = df.head(args.limit)
    df = df.reset_index(drop=True)

    docs_jsonl = out_dir / "documents.jsonl"
    articles_jsonl = out_dir / "articles.jsonl"
    failed_csv = out_dir / "failed.csv"
    r2ai_dir = out_dir / "r2ai"
    r2ai_json_dir = r2ai_dir / "filtered" / "json"
    r2ai_chunks = {}
    if not args.no_r2ai:
        r2ai_json_dir.mkdir(parents=True, exist_ok=True)

    failed_rows = []
    with docs_jsonl.open("w", encoding="utf-8") as f_doc, articles_jsonl.open("w", encoding="utf-8") as f_art:
        row_dicts = [row.to_dict() for _, row in df.iterrows()]

        def handle_result(row_dict: dict, result: dict) -> None:
            doc_record = {k: v for k, v in result.items() if k != "articles"}
            f_doc.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

            if not args.no_r2ai:
                r2ai_doc = build_r2ai_document(result)
                (r2ai_json_dir / f"{safe_filename(result['id'])}.json").write_text(
                    json.dumps(r2ai_doc, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            if not result["ok"]:
                failed_rows.append(row_dict)
                return

            for art in result["articles"]:
                art_record = {
                    "doc_id": result["id"],
                    "so_ky_hieu": result["so_ky_hieu"],
                    "ten_van_ban": result["ten_van_ban"],
                    "loai_van_ban": result["loai_van_ban"],
                    "source_url": result["source_url"],
                    **art,
                }
                f_art.write(json.dumps(art_record, ensure_ascii=False) + "\n")
                if not args.no_r2ai:
                    chunk_key = f"{art.get('chunk_level') or 'article'}_{art.get('chunk_no') or art.get('article_no') or '0'}"
                    key = unique_article_key(r2ai_chunks, result["id"], chunk_key)
                    r2ai_chunks[key] = build_r2ai_article_chunk(result, art)

        if args.workers <= 1:
            for row_dict in tqdm(row_dicts, total=len(row_dicts)):
                result = crawl_one(row_dict, out_dir)
                handle_result(row_dict, result)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(crawl_one, row_dict, out_dir): row_dict for row_dict in row_dicts}
                for future in tqdm(as_completed(futures), total=len(futures)):
                    row_dict = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        failed_rows.append({**row_dict, "error": str(exc)})
                        continue
                    handle_result(row_dict, result)

    if not args.no_r2ai:
        (r2ai_dir / "articles.json").write_text(
            json.dumps(r2ai_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(failed_csv, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Documents: {docs_jsonl}")
    print(f"Articles:  {articles_jsonl}")
    if not args.no_r2ai:
        print(f"R2AI docs: {r2ai_json_dir}")
        print(f"R2AI chunks: {r2ai_dir / 'articles.json'}")
    print(f"Failed:    {failed_csv if failed_rows else 'none'}")


if __name__ == "__main__":
    main()

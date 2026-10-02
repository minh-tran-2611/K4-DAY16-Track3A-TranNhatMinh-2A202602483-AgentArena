"""Bằng chứng dùng chung cho `critic` và `citation_checker`.

Hai lớp đó cùng hỏi một câu: "câu trích này nằm NGUYÊN VĂN trong MỘT DÒNG
của tài liệu nào mà lượt chạy đã thực sự lấy về?" Câu hỏi được trả lời
đúng theo luật của `arena/scorer.py` (`_norm` + `_supports`): so sánh sau
khi chuẩn hoá NFC, casefold, gộp khoảng trắng, và chỉ trong phạm vi MỘT
dòng. Đặt ở một chỗ để hai lớp không bao giờ lệch nhau.

"Đã lấy về" được ghi lại ở `wrap_tool_call` (`note_tool_call`): mọi
`fetch_doc` đã thử, cộng mọi doc_id xuất hiện trong kết quả `search`. Đó
là tập con của `retrieved` mà scorer dựng lại từ trace, nên gắn claim vào
một tài liệu trong tập này không bao giờ bị chấm `UNRETRIEVED`.

Không đọc `Doc.tags` (luôn rỗng) và không đọc brief.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")
_DOC_ID_RE = re.compile(r"doc-\d{4}")

#: Như `arena.scorer.MIN_SUPPORT_CHARS`: ngắn hơn thì scorer không công nhận.
MIN_SUPPORT_CHARS = 12

#: Ký tự được phép CẮT ở hai đầu claim (cắt là hợp lệ, xem README §8.1).
#: Mô hình thật hay chép cả dấu "…" của snippet hoặc thêm dấu chấm cuối.
EDGE_CHARS = " \t\r\n.,;:…\"'“”‘’«»()[]*-–—"

_SEEN_KEY = "_evidence_seen_docs"
_LINES_KEY = "_evidence_doc_lines"


def norm(text) -> str:
    """Cùng phép chuẩn hoá với `arena.scorer._norm`."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def note_tool_call(ctx, name, args, result) -> None:
    """Ghi lại doc_id mà một lượt gọi công cụ đã đưa vào lượt chạy."""
    seen = ctx.state.setdefault(_SEEN_KEY, [])
    found = []
    if name == "fetch_doc" and isinstance(args, dict):
        doc_id = args.get("doc_id")
        if isinstance(doc_id, str):
            found.append(doc_id.strip())
    elif name == "search" and getattr(result, "ok", False):
        found.extend(_DOC_ID_RE.findall(getattr(result, "content", "") or ""))
    for doc_id in found:
        if doc_id not in seen:
            seen.append(doc_id)


def seen_doc_ids(ctx) -> list:
    return list(ctx.state.get(_SEEN_KEY, []))


def _doc_lines(ctx, doc_id: str) -> tuple:
    cache = ctx.state.setdefault(_LINES_KEY, {})
    if doc_id not in cache:
        doc = ctx.corpus.get(doc_id) if ctx.corpus is not None else None
        body = doc.body if doc is not None else ""
        cache[doc_id] = tuple(l for l in (norm(r) for r in body.splitlines()) if l)
    return cache[doc_id]


def supports(ctx, doc_id, text) -> bool:
    """Tài liệu `doc_id` có chứa `text` nguyên văn trong MỘT dòng không?"""
    n = norm(text)
    if len(n) < MIN_SUPPORT_CHARS or not isinstance(doc_id, str):
        return False
    return any(n in line for line in _doc_lines(ctx, doc_id))


def source_of(ctx, text, prefer=None):
    """doc_id đã lấy về đầu tiên chứa `text` trong một dòng, hoặc None.

    `prefer` (doc_id mà claim đang trích) được thử trước, để một trích dẫn
    đã đúng không bị chuyển sang tài liệu khác chỉ vì cùng có câu đó.
    """
    seen = seen_doc_ids(ctx)
    candidates = ([prefer] if prefer in seen else []) + seen
    for doc_id in candidates:
        if supports(ctx, doc_id, text):
            return doc_id
    return None


def trimmed(text: str) -> str:
    """`text` bỏ ký tự rác ở hai đầu — vẫn là một substring của nó."""
    return text.strip(EDGE_CHARS) if isinstance(text, str) else ""


def sync_citations(report: dict) -> None:
    """`citations` = các doc_id mà claims còn lại thực sự trích."""
    ids = set()
    for claim in report.get("claims") or []:
        if isinstance(claim, dict) and isinstance(claim.get("doc_id"), str) and claim["doc_id"]:
            ids.add(claim["doc_id"])
    report["citations"] = sorted(ids)

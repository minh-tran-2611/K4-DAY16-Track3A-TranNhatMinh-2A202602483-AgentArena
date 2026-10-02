"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from arena.model import MockModel

from harness.layers._evidence import note_tool_call, source_of, sync_citations
from harness.middleware import Middleware

#: Nối vào SYSTEM message khi chạy model thật. Đo trên gemini-3.6-flash:
#: model chép đúng một câu nhưng dừng ở dấu chấm giữa dòng, nên phủ 5/11
#: từ khoá của dữ kiện (< 60%) và recall = 0 dù claim SUPPORTED. Layer
#: không được nối thêm chữ vào claim (NOT_FROM_MODEL), nên phải nói trước.
QUOTE_GUIDANCE = (
    "\n\nHƯỚNG DẪN TRÍCH DẪN: mỗi phần tử claims hãy chép TRỌN VẸN cả một dòng "
    "của tài liệu chứa câu trả lời (từ đầu dòng tới cuối dòng, kể cả các câu "
    "đứng sau dấu chấm trong cùng dòng đó), đúng từng ký tự, miễn là dưới 400 "
    "ký tự. Không dừng ở giữa dòng."
)


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def before_model(self, ctx, messages):
        # Mock trích sẵn trọn dòng; giữ đường mock byte-identical (token).
        if isinstance(getattr(ctx.model, "inner", ctx.model), MockModel):
            return messages
        if not messages or messages[0].get("role") != "system":
            return messages
        # Bản sao mới: không dán vĩnh viễn vào lịch sử của agent.
        system = dict(messages[0], content=str(messages[0].get("content", "")) + QUOTE_GUIDANCE)
        return [system] + messages[1:]

    def wrap_tool_call(self, ctx, call, name, args):
        result = call(name, args)
        note_tool_call(ctx, name, args, result)  # tài liệu nào đã thực sự vào lượt chạy
        return result

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report
        moved = 0
        for claim in claims:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
                continue
            cited = claim.get("doc_id")
            # `source_of` thử chính `cited` trước: trích đúng rồi thì giữ.
            source = source_of(ctx, claim["text"], cited)
            if source is not None and source != cited:
                claim["doc_id"] = source  # GẮN LẠI — chữ giữ nguyên
                moved += 1
            # Không tìm được nguồn: đó là bịa, để `critic` xoá.
        ctx.state["citations_moved"] = moved
        sync_citations(report)
        return report

# Báo cáo Phân tích Thất bại (Failure Analysis Report)

> Được sinh tự động từ thư mục `reports/` bằng script `analysis/generate_analysis.py` — phiên bản **V2_Optimized**, cập nhật lúc 2026-06-16 22:27:20.

## 1. Tổng quan Benchmark
- **Tổng số cases:** 60
- **Số case Pass / Fail:** 57 / 3  (tương ứng pass rate 95.0%)
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.978
    - Answer Relevancy: 1.000
- **Tầng Retrieval:** Hit Rate 98.2% · MRR 0.973
- **Điểm LLM-Judge trung bình:** 4.66 / 5.0
- **Độ tin cậy giám khảo:** Agreement 98.3% · Cohen's Kappa 0.6021 · 3 ca xảy ra xung đột
- **Position Bias (sau audit):** 16.7%
- **Hiệu năng và Chi phí:** tổng 54.92s · $0.000921/eval · tổng $0.055266

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Tỉ lệ trên tổng fail |
|----------|:--------:|:--------------------:|
| Other (điểm thấp do nguyên nhân khác) | 1 | 33% |
| Judge Conflict (giám khảo bất đồng, chất lượng mơ hồ) | 1 | 33% |
| Retrieval Miss (truy xuất sai hoặc không truy xuất được) | 1 | 33% |

**Phân bố fail theo loại câu hỏi:** `fact-check`×2, `ambiguous`×1

## 3. Phân tích 5 Whys (3 case tệ nhất)

### Case #1: `ADV-AMB-02` — ambiguous (điểm 1.0/5)
- **Câu hỏi:** Cái thẻ đó cấp lại mất bao nhiêu tiền?
- **Agent trả lời:** Tôi không tìm thấy thông tin này trong tài liệu nội bộ. Bạn vui lòng liên hệ bộ phận phụ trách để được hỗ trợ.
- **Đáp án chuẩn:** Cần làm rõ "thẻ" nào (thẻ ra vào hay loại thẻ khác). Nếu là thẻ ra vào thì phí là 100.000đ (OPS-002).

**Triệu chứng:** Agent phản hồi rằng không tìm thấy thông tin, không cung cấp được đáp án chuẩn.

*   **Why 1:** Agent không tìm thấy thông tin liên quan trong ngữ cảnh được cung cấp.
*   **Why 2:** Tài liệu `OPS-002` (chứa thông tin phí thẻ ra vào) đã không được truy xuất. (`hit_rate=0.0`)
*   **Why 3:** Hệ thống retriever không chọn được tài liệu phù hợp nhất là `OPS-002` cho câu hỏi này.
*   **Why 4:** Câu hỏi của người dùng quá chung chung ("cái thẻ đó") hoặc embedding của câu hỏi chưa đủ mạnh để khớp với các đoạn tài liệu cụ thể về "thẻ ra vào" trong `OPS-002`.
*   **Root Cause:** Hệ thống truy xuất chưa được tối ưu để xử lý các câu hỏi mơ hồ hoặc thiếu ngữ cảnh, dẫn tới việc không tìm ra những tài liệu có chứa từ khóa cụ thể hơn (ví dụ: "thẻ ra vào").

**Lỗi nằm ở tầng:** Retrieval

### Case #2: `G-009` — fact-check (điểm 3.0/5)
- **Câu hỏi:** Khi làm việc từ xa thì truy cập tài nguyên nội bộ bằng cách nào?
- **Agent trả lời:** Khi làm việc từ xa, bạn dùng GlobalProtect VPN để truy cập tài nguyên nội bộ.
- **Đáp án chuẩn:** Dùng GlobalProtect VPN, đăng nhập bằng tài khoản công ty kèm 2FA.

**Triệu chứng:** Câu trả lời thiếu thông tin quan trọng về cách đăng nhập (2FA, tài khoản công ty).
**Why 1:** Agent không tổng hợp đầy đủ chi tiết từ ngữ cảnh đã truy xuất.
**Why 2:** Mô hình ngôn ngữ (LLM) không trích xuất hoặc ưu tiên các chi tiết phụ (dù quan trọng) từ ngữ cảnh.
**Why 3:** Hướng dẫn trong prompt (system prompt hoặc user prompt) chưa đủ mạnh để yêu cầu LLM cung cấp *tất cả* các bước hoặc chi tiết cần thiết.
**Why 4:** Prompt được thiết kế chưa tối ưu để đảm bảo tính đầy đủ, có xu hướng ưu tiên sự ngắn gọn mà không có chỉ dẫn rõ ràng về việc liệt kê đầy đủ các bước.
**Root Cause:** Lỗi nằm ở khâu thiết kế prompt, không yêu cầu LLM trích xuất và tổng hợp đầy đủ các chi tiết quan trọng từ ngữ cảnh đã được truy xuất.

**Lỗi nằm ở tầng:** Prompting

### Case #3: `G-028` — fact-check (điểm 3.0/5)
- **Câu hỏi:** Ngày thường tôi cần mặc trang phục gì đi làm?
- **Agent trả lời:** Vào các ngày trong tuần, bạn cần mặc trang phục công sở lịch sự (smart casual).
- **Đáp án chuẩn:** Trang phục công sở lịch sự (smart casual); riêng thứ Sáu được phép mặc casual nhưng vẫn cần gọn gàng.

**Triệu chứng:** Câu trả lời thiếu thông tin về quy định trang phục vào ngày thứ Sáu.

*   **Why 1:** LLM không tổng hợp đầy đủ thông tin từ ngữ cảnh được truy xuất.
*   **Why 2:** LLM đã bỏ sót một phần thông tin chi tiết (quy định thứ Sáu) có trong ngữ cảnh.
*   **Why 3:** Prompt không yêu cầu LLM phải trích xuất hoặc tổng hợp *tất cả* các quy tắc/chi tiết liên quan.
*   **Why 4:** Prompt còn chung chung, thiếu hướng dẫn cụ thể để đảm bảo câu trả lời đầy đủ chi tiết, đặc biệt với các trường hợp có ngoại lệ.
*   **Root Cause:** Quy trình thiết kế và kiểm thử prompt chưa đủ chặt chẽ để bao quát các trường hợp có ngoại lệ hoặc yêu cầu chi tiết toàn diện.

**Lỗi nằm ở tầng:** Prompting

## 4. Hiệu năng & Chi phí (Performance & Cost)
- **Thời gian:** 54.92s cho 60 case (trung bình khoảng 1.576s/case) — chạy song song theo cơ chế async.
- **Token và chi phí:** 115.562 tokens · 247 calls · tổng **$0.055266** (trung bình **$0.000921/eval**).
- **Phân bổ theo model:**
    - `gemini-2.5-flash`: 184 calls · $0.051321
    - `gemini-2.5-flash-lite`: 63 calls · $0.003945

**Đề xuất giảm khoảng 30% chi phí eval (giữ nguyên độ chính xác) — Judge Cascade:**
1. Ở vòng đầu chỉ chạy **1 judge rẻ** (`flash-lite`) cho tất cả các case.
2. Nếu điểm rơi vào vùng rõ ràng (1–2 hoặc 5) và chỉ số faithfulness cao → **chốt ngay kết quả**, bỏ qua judge thứ 2.
3. Chỉ **escalate** lên judge 2 kèm cơ chế debate cho những case có điểm lưng chừng (3–4) hoặc nghi ngờ xung đột.
Do phân bố điểm lệch mạnh về phía 4–5, ước tính hơn 40% case chỉ cần 1 lần gọi → giảm khoảng 30–40% chi phí giám khảo. Có thể bổ sung thêm **caching** cho những câu hỏi trùng lặp để tiết kiệm thêm.

## 5. Kế hoạch cải tiến (Action Plan)
_Ưu tiên dựa trên nhóm lỗi lớn nhất: **Other (điểm thấp, nguyên nhân khác)**._

- [ ] Bổ sung **embedding-based retrieval kết hợp reranking** thay cho lexical thuần nhằm giảm Retrieval Miss.
- [ ] Tăng cường **system prompt chống hallucination** (chỉ trả lời theo context, từ chối khi thiếu thông tin).
- [ ] Thêm **few-shot guardrail** cho các mẫu red-team (prompt injection, out-of-context).
- [ ] Thử nghiệm **chunking theo ngữ nghĩa (semantic chunking)** thay cho fixed-size.
- [ ] Mở rộng golden set ở những loại câu hỏi đang fail nhiều để tinh chỉnh agent có trọng tâm hơn.

---
_Bản nháp được sinh tự động — nhóm cần rà soát và bổ sung nhận định chuyên môn trước khi nộp._

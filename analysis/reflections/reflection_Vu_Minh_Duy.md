# Báo cáo Cá nhân (Individual Reflection)

> **Họ tên:** Vũ Minh Duy
> **MSSV:** 2A202600806
> **Vai trò trong nhóm:** AI/Backend — Multi-Judge Consensus Engine & Async Runner

---

## 1. Đóng góp kỹ thuật (Engineering Contribution — 15đ)

Trong dự án, tôi đảm nhận phần xây dựng các module cốt lõi và phức tạp nhất của hệ thống, bao gồm **cơ chế hội đồng giám khảo đa mô hình**, **bộ chạy benchmark bất đồng bộ**, và **lớp kết nối với LLM**.

### a) `engine/llm_judge.py` — Multi-Judge Consensus Engine
- Tôi xây dựng class `LLMJudge` cho phép một giám khảo đơn lẻ chấm điểm theo 4 tiêu chí rõ ràng (tính chính xác, tính đầy đủ, sự chuyên nghiệp, độ an toàn) trên thang điểm 1–5, kết quả trả về dạng JSON có xác thực hợp lệ.
- Tiếp đến, tôi phát triển `MultiModelJudge` cho phép **2 mô hình giám khảo khác nhau chấm điểm đồng thời** (`gemini-2.5-flash` kết hợp `gemini-2.5-flash-lite`).
- Tôi thiết kế **quy trình leo thang xử lý xung đột (escalation ladder)** thay vì chỉ đơn giản lấy trung bình cộng:
  1. **Chấm độc lập** — Hai giám khảo làm việc song song; nếu khoảng cách điểm ≤1 thì lấy trung bình.
  2. **Vòng tranh luận (debate)** — Gọi hàm `revise()`: mỗi bên đọc điểm và lý do của bên còn lại, sau đó chấm lại. Nếu hai bên đồng thuận thì dừng ngay, giúp tiết kiệm một lượt gọi trọng tài.
  3. **Trọng tài (arbiter)** — Trường hợp vẫn bất đồng, một mô hình thứ 3 sẽ phân xử và kết quả lấy theo **median**.
- Bổ sung hàm **`cohens_kappa()`** để đo lường mức độ nhất quán giữa hai giám khảo trên toàn bộ tập dữ liệu.
- Tôi cũng hiện thực **`check_position_bias()`** để phát hiện hiện tượng thiên vị vị trí bằng cách hoán đổi vị trí A/B giữa các lần chấm.

### b) `engine/runner.py` — Async Benchmark Runner
- Sử dụng `asyncio.Semaphore` để giới hạn số lượng tác vụ chạy đồng thời, tránh tình trạng vượt rate-limit của API.
- Bên trong mỗi case, RAGAS và Multi-Judge được kích hoạt **song song** thông qua `asyncio.gather`.
- Hệ thống ghi nhận **độ trễ, số token sử dụng, và chi phí (USD)** thông qua `usage_ledger`, từ đó tính ra chỉ số `cost_per_eval`.
- Đầu ra tổng hợp bao gồm Hit Rate, MRR, Cohen's Kappa, tỉ lệ đồng thuận, và số ca xảy ra xung đột.
- Mỗi case được **cô lập lỗi** với timeout `asyncio.wait_for` riêng, đảm bảo một case hỏng không kéo sập toàn bộ benchmark.

### c) `engine/llm_client.py` — Lớp client thống nhất
- Kết nối tới Gemini thông qua endpoint tương thích OpenAI, đồng thời có cơ chế **fallback offline mock** cho trường hợp chưa có API key.
- Theo dõi tổng token và chi phí ở cấp toàn cục.
- Phân biệt rõ xử lý retry/backoff cho lỗi rate-limit (HTTP 429) và đặt `timeout=30s` cho client (chi tiết ở mục Problem Solving).

**Bằng chứng:** lịch sử commit cùng các file tương ứng (`git log --oneline`, `git blame engine/llm_judge.py`).

---

## 2. Chiều sâu kỹ thuật (Technical Depth — 15đ)

### MRR (Mean Reciprocal Rank)
Với mỗi truy vấn, **reciprocal rank = 1 / (vị trí xuất hiện ĐẦU TIÊN của tài liệu đúng)**; nếu không có tài liệu đúng nào, giá trị này bằng 0. MRR là giá trị trung bình cộng của các reciprocal rank trên toàn bộ tập truy vấn.
Điểm khác biệt so với **Hit Rate** (chỉ kiểm tra nhị phân tài liệu đúng có nằm trong top-k hay không) là MRR còn **khuyến khích đẩy tài liệu đúng lên vị trí cao**: vị trí 1 được 1.0 điểm, vị trí 2 được 0.5, vị trí 3 được 0.33. Trong hệ thống của tôi, Hit Rate đạt 98.2% nhưng MRR chỉ 0.973 (<Hit Rate), cho thấy vẫn còn những case tài liệu đúng bị đẩy xuống dưới vị trí đầu — đây là dấu hiệu cho thấy cần bổ sung cơ chế reranking.

### Cohen's Kappa
Công thức: **κ = (Po − Pe) / (1 − Pe)**, trong đó Po là tỉ lệ đồng thuận thực tế quan sát được, còn Pe là tỉ lệ đồng thuận **kỳ vọng nếu chỉ do ngẫu nhiên quyết định**.
Kappa ưu việt hơn Agreement Rate thuần vì nó **loại bỏ phần đồng thuận do may rủi**: khi cả hai giám khảo đều quen cho điểm 4–5, Agreement Rate sẽ phóng đại mức độ nhất quán, trong khi Kappa trừ đi đúng thành phần ngẫu nhiên đó.
Kết quả benchmark cho thấy: V1 đạt κ=0.611, V2 đạt κ=0.526 (mức **moderate** theo thang Landis–Koch), nghĩa là hai giám khảo nhất quán ở mức chấp nhận được sau khi đã loại bỏ yếu tố may rủi.

### Position Bias
LLM-judge thường có xu hướng **thiên vị câu trả lời ở một vị trí cố định** (thường là vị trí đầu tiên) bất kể chất lượng thực tế. Tôi phát hiện điều này bằng cách đưa cùng một cặp (A, B) theo **hai thứ tự khác nhau**, rồi quan sát lựa chọn "câu nào tốt hơn" có **thay đổi** khi hoán đổi vị trí hay không. Nếu có sự thay đổi thì giám khảo đang bị bias.
Kết quả audit cho thấy **position_bias_rate = 16.7%**, khẳng định giám khảo chưa thật sự khách quan, củng cố thêm lý do phải kết hợp nhiều judge cùng cơ chế debate thay vì dựa vào một mô hình duy nhất.

### Cân bằng giữa Chi phí và Chất lượng
- Phiên bản V2 cải thiện **+0.16 điểm** và tăng pass rate thêm 4% trong khi chi phí chỉ nhân **1.09×** so với V1 — một sự cải thiện gần như "không tốn thêm".
- Cơ chế debate/arbiter chỉ kích hoạt khi **xung đột thật sự** (3/60 ca), nên chi phí biên rất thấp, không phải lúc nào cũng phải gọi tới 3 mô hình.
- Lựa chọn `gemini-2.5-flash-lite` làm judge thứ 2 (rẻ hơn flash khoảng 3 lần) thay vì dùng bản pro giúp **đa dạng hóa mô hình mà vẫn tiết kiệm**.
- Chi phí thực đo: **$0.00093 / eval** (tương đương khoảng $0.055 cho toàn bộ 60 case).

---

## 3. Giải quyết vấn đề (Problem Solving — 10đ)

**Vấn đề:** Benchmark async **bị treo lặp đi lặp lại** ở khoảng case 50/60, chạy hơn 7 phút vẫn chưa xong mà không có thông báo lỗi cụ thể.

**Quá trình chẩn đoán theo 3 tầng nguyên nhân:**
1. **Tiến trình zombie:** Phát hiện có tới 5 tiến trình `python.exe` cùng hoạt động. Các lần chạy trước đã bị dừng nhưng tiến trình con **vẫn sống**, liên tục ngốn quota API và gây rate-limit chéo lẫn nhau. → Tôi tiến hành kill sạch toàn bộ tiến trình.
2. **Arbiter phản hồi chậm:** Vùng bị treo (case 51–60) đúng là nhóm case **red-team** thường xuyên gây xung đột, dẫn tới việc kích hoạt gọi `gemini-2.5-pro` — một mô hình có **RPM rất thấp** — nên thời gian backoff kéo dài. → Tôi đổi arbiter mặc định sang `gemini-2.5-flash`.
3. **Nguyên nhân GỐC RỄ:** `AsyncOpenAI` mặc định có **timeout = 600 giây**. Khi một request bị treo, cả case đó sẽ bị chặn tới 10 phút. → Tôi đặt `timeout=30s, max_retries=0` (tự quản lý retry), bổ sung `asyncio.wait_for` timeout 90s/case, và cô lập lỗi cho từng case.

**Kết quả:** Sau khi sửa, benchmark chạy gọn trong **296 giây** cho 60 case × 2 phiên bản (V1 mất 150s, V2 mất 114s — đạt mục tiêu <2 phút/50 case của V2), sinh report đầy đủ và quyết định cuối cùng là **APPROVE**.

**Bài học rút ra:** Trong hệ thống async, **timeout mặc định của thư viện** còn nguy hiểm hơn cả lỗi rõ ràng, vì nó khiến tiến trình "treo im lặng". Luôn cần đặt timeout tường minh và cô lập lỗi ở cấp độ từng đơn vị công việc.

---

## 4. Đề xuất tối ưu chi phí 30% (Bonus)

Rubric yêu cầu một đề xuất giảm **30% chi phí eval mà vẫn giữ nguyên độ chính xác**. Phương án tôi đề xuất là **Judge Cascade (giám khảo phân tầng)**:

1. **Vòng 1 — chạy giám khảo rẻ:** Chỉ sử dụng **1 judge rẻ** (`flash-lite`) để chấm tất cả các case.
2. **Bộ lọc độ tự tin:** Nếu điểm rơi vào vùng "rõ ràng" (1–2 hoặc 5) đồng thời chỉ số faithfulness cao → **chốt kết quả luôn**, bỏ qua judge thứ 2.
3. **Chỉ escalate** lên judge thứ 2 kèm debate cho những case có **điểm lưng chừng (3–4)** hoặc nghi ngờ xung đột.

Do phân bố điểm thực tế lệch mạnh về phía 4–5 (đa số case là dạng dễ), ước tính **hơn 40% case chỉ cần 1 lần gọi thay vì 2**, từ đó giảm được khoảng **30–40% chi phí giám khảo** mà không ảnh hưởng tới các case khó — vốn là những case thật sự cần độ chính xác cao. Ngoài ra, có thể bổ sung **caching kết quả** cho những câu hỏi trùng lặp để tiết kiệm thêm.

## 5. Hạn chế & hướng phát triển
- Retriever hiện tại dùng **TF-IDF lexical**, yếu trong việc xử lý khác biệt từ vựng (gây ra 1 ca Retrieval Miss). Hướng cải tiến: **chuyển sang embedding-based retrieval kết hợp reranking**.
- Tỉ lệ position bias 16.7% vẫn còn đáng kể → có thể bổ sung bước **calibration** để chuẩn hóa phân phối điểm giữa các judge.
- Golden set nên được mở rộng ở nhóm câu hỏi đang fail nhiều (ambiguous, multi-hop) để tinh chỉnh agent một cách có trọng tâm hơn.

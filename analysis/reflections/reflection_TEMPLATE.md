# Báo cáo Cá nhân (Individual Reflection)

> **Họ tên:** _[Điền tên]_
> **Vai trò trong nhóm:** _[VD: Data/SDG · AI-Backend/Multi-Judge · DevOps/Regression · Frontend]_

---

## 1. Đóng góp kỹ thuật (Engineering Contribution — 15đ)
Liệt kê cụ thể module/đoạn code bạn đóng góp (kèm commit nếu có):

- **Module phụ trách:** _[VD: engine/llm_judge.py — Multi-Judge consensus + debate]_
- **Việc đã làm:**
  - _[VD: Triển khai Cohen's Kappa và logic escalation độc lập → debate → arbiter]_
  - _[...]_
- **Bằng chứng (Git commit / PR):** _[hash hoặc link]_

## 2. Chiều sâu kỹ thuật (Technical Depth — 15đ)
Giải thích bằng lời của bạn (không copy):

- **MRR là gì, tính thế nào, khác Hit Rate ra sao?**
  _[Trả lời...]_
- **Cohen's Kappa đo gì? Vì sao tốt hơn Agreement Rate thuần?**
  _[Gợi ý: Kappa loại bỏ phần đồng thuận do may rủi: (Po−Pe)/(1−Pe)...]_
- **Position Bias là gì? Hệ thống kiểm tra thế nào?**
  _[Gợi ý: đảo vị trí A/B, xem giám khảo có đổi lựa chọn không...]_
- **Trade-off Chi phí ↔ Chất lượng:** Bạn rút ra gì từ số liệu cost/eval và việc dùng arbiter/debate?
  _[Trả lời...]_

## 3. Giải quyết vấn đề (Problem Solving — 10đ)
Một vấn đề khó bạn gặp khi xây hệ thống phức tạp và cách bạn xử lý:

- **Vấn đề:** _[VD: Rate-limit của Gemini free-tier làm benchmark async chậm/đứng]_
- **Cách chẩn đoán:** _[...]_
- **Giải pháp:** _[VD: giảm concurrency, backoff theo 429, fallback arbiter...]_
- **Kết quả:** _[...]_

## 4. Bài học & Hướng cải tiến
- _[Nếu làm lại, bạn sẽ thay đổi gì?]_
- _[Hệ thống còn điểm yếu nào? Bạn đề xuất gì?]_

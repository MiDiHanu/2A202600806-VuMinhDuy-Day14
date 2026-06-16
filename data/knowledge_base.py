"""
Knowledge Base (corpus) cho Lab 14 — Sổ tay hỗ trợ nội bộ công ty giả định "TechNova".

Mỗi chunk có `id` duy nhất => dùng làm Ground Truth cho Retrieval Eval (Hit Rate / MRR).
Agent RAG sẽ retrieve trên chính corpus này, nên `expected_retrieval_ids` trong
golden set là CÓ THẬT và đo được, không phải bịa.

Chạy trực tiếp để ghi ra data/corpus.jsonl:
    python data/knowledge_base.py
"""
import io
import sys
import json
import os
from typing import Dict, List

# Đảm bảo in được tiếng Việt/emoji trên Windows console (cp1252).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Mỗi phần tử: {id, category, title, text}
CORPUS: List[Dict] = [
    # ---------------- IT / Tài khoản ----------------
    {
        "id": "IT-001", "category": "Tài khoản",
        "title": "Đổi mật khẩu",
        "text": "Để đổi mật khẩu, truy cập portal.technova.vn, đăng nhập, vào mục 'Bảo mật' rồi chọn 'Đổi mật khẩu'. Mật khẩu mới phải dài tối thiểu 12 ký tự, gồm chữ hoa, chữ thường, số và ký tự đặc biệt. Hệ thống bắt buộc đổi mật khẩu mỗi 90 ngày.",
    },
    {
        "id": "IT-002", "category": "Tài khoản",
        "title": "Quên mật khẩu / khóa tài khoản",
        "text": "Nếu nhập sai mật khẩu 5 lần liên tiếp, tài khoản sẽ bị khóa trong 30 phút. Để mở khóa ngay, dùng chức năng 'Quên mật khẩu' tại trang đăng nhập hoặc gọi IT Helpdesk số nội bộ 1800. Link đặt lại mật khẩu có hiệu lực trong 15 phút.",
    },
    {
        "id": "IT-003", "category": "Bảo mật",
        "title": "Xác thực hai lớp (2FA)",
        "text": "Toàn bộ nhân viên bắt buộc bật xác thực hai lớp (2FA) qua ứng dụng Microsoft Authenticator. SMS OTP không được chấp nhận vì lý do bảo mật. Nếu mất thiết bị 2FA, phải báo IT Helpdesk trong vòng 1 giờ để vô hiệu hóa thiết bị cũ.",
    },
    {
        "id": "IT-004", "category": "Thiết bị",
        "title": "Yêu cầu cấp laptop mới",
        "text": "Nhân viên chính thức được cấp 1 laptop. Để yêu cầu thay/cấp mới, tạo ticket trên hệ thống ITSM, mục 'Hardware Request', cần có phê duyệt của quản lý trực tiếp. Thời gian xử lý trung bình là 5 ngày làm việc.",
    },
    {
        "id": "IT-005", "category": "Thiết bị",
        "title": "Cài đặt phần mềm",
        "text": "Chỉ phần mềm trong 'Software Center' mới được tự cài. Phần mềm ngoài danh sách phải được Bộ phận An ninh thông tin (InfoSec) duyệt qua ticket. Tuyệt đối không cài phần mềm crack; vi phạm sẽ bị xử lý kỷ luật.",
    },
    {
        "id": "IT-006", "category": "Mạng",
        "title": "VPN truy cập từ xa",
        "text": "Khi làm việc từ xa, dùng GlobalProtect VPN để truy cập tài nguyên nội bộ. Đăng nhập VPN bằng tài khoản công ty kèm 2FA. VPN tự ngắt sau 12 giờ và cần đăng nhập lại. Không chia sẻ kết nối VPN cho thiết bị cá nhân.",
    },
    {
        "id": "IT-007", "category": "Email",
        "title": "Dung lượng & lưu trữ email",
        "text": "Hộp thư mỗi nhân viên có dung lượng 50GB. Khi đầy 90%, hệ thống gửi cảnh báo. Email cũ hơn 2 năm được tự động lưu trữ (archive) và vẫn tìm kiếm được qua mục Online Archive.",
    },
    {
        "id": "IT-008", "category": "Bảo mật",
        "title": "Báo cáo email lừa đảo (phishing)",
        "text": "Nếu nghi ngờ email lừa đảo, KHÔNG bấm vào link hay tải tệp đính kèm. Dùng nút 'Report Phishing' trên Outlook để báo cáo. InfoSec sẽ phản hồi trong vòng 4 giờ làm việc.",
    },
    # ---------------- HR / Nhân sự ----------------
    {
        "id": "HR-001", "category": "Nghỉ phép",
        "title": "Số ngày phép năm",
        "text": "Nhân viên chính thức có 12 ngày phép năm có lương. Sau mỗi 3 năm thâm niên được cộng thêm 1 ngày, tối đa 18 ngày. Phép năm không dùng hết được chuyển tối đa 5 ngày sang năm sau, hạn dùng đến 31/03.",
    },
    {
        "id": "HR-002", "category": "Nghỉ phép",
        "title": "Quy trình xin nghỉ phép",
        "text": "Đơn xin nghỉ phép nộp qua hệ thống HRM ít nhất 3 ngày làm việc trước ngày nghỉ. Nghỉ trên 3 ngày liên tục cần phê duyệt của trưởng phòng. Nghỉ đột xuất do ốm phải báo quản lý trước 9h sáng cùng ngày.",
    },
    {
        "id": "HR-003", "category": "Nghỉ phép",
        "title": "Nghỉ ốm và giấy tờ y tế",
        "text": "Nghỉ ốm từ 2 ngày trở lên phải nộp giấy chứng nhận của cơ sở y tế. Nghỉ ốm có lương tối đa 30 ngày/năm theo quy định bảo hiểm xã hội. Giấy tờ nộp trong vòng 5 ngày sau khi đi làm lại.",
    },
    {
        "id": "HR-004", "category": "Lương thưởng",
        "title": "Chu kỳ trả lương",
        "text": "Lương được trả vào ngày 5 hàng tháng cho tháng làm việc trước đó. Nếu ngày 5 rơi vào cuối tuần hoặc ngày lễ, lương được trả vào ngày làm việc liền trước. Phiếu lương xem trên cổng HRM.",
    },
    {
        "id": "HR-005", "category": "Lương thưởng",
        "title": "Thưởng hiệu suất",
        "text": "Thưởng hiệu suất được đánh giá 2 lần/năm (tháng 6 và tháng 12), dựa trên kết quả OKR cá nhân và kết quả kinh doanh của công ty. Mức thưởng dao động 0–3 tháng lương tùy xếp loại.",
    },
    {
        "id": "HR-006", "category": "Phúc lợi",
        "title": "Bảo hiểm sức khỏe",
        "text": "Công ty cung cấp gói bảo hiểm sức khỏe PVI cho nhân viên chính thức, hiệu lực sau khi hết thử việc. Người thân (vợ/chồng, con) có thể mua thêm với mức phí ưu đãi, đăng ký trong tháng 1 hàng năm.",
    },
    {
        "id": "HR-007", "category": "Làm việc",
        "title": "Chính sách làm việc từ xa (hybrid)",
        "text": "Công ty áp dụng mô hình hybrid: tối đa 2 ngày làm từ xa mỗi tuần, đăng ký lịch với quản lý vào đầu tuần. Một số vị trí đặc thù (vận hành, lễ tân) bắt buộc làm tại văn phòng toàn thời gian.",
    },
    {
        "id": "HR-008", "category": "Làm việc",
        "title": "Giờ làm việc & chấm công",
        "text": "Giờ làm việc tiêu chuẩn là 8h30–17h30, nghỉ trưa 1 tiếng, từ thứ Hai đến thứ Sáu. Chấm công bằng thẻ từ hoặc app TechNova Attendance. Đi muộn quá 3 lần/tháng sẽ bị nhắc nhở.",
    },
    {
        "id": "HR-009", "category": "Onboarding",
        "title": "Thử việc",
        "text": "Thời gian thử việc là 2 tháng đối với nhân viên và 2 tháng đối với cấp quản lý, hưởng 85% lương chính thức. Kết thúc thử việc có buổi đánh giá; nếu đạt sẽ ký hợp đồng chính thức.",
    },
    {
        "id": "HR-010", "category": "Chi phí",
        "title": "Hoàn ứng công tác phí",
        "text": "Chi phí công tác (đi lại, khách sạn, ăn uống) được hoàn theo hóa đơn hợp lệ. Nộp đề nghị hoàn ứng trên hệ thống Finance trong vòng 14 ngày sau chuyến đi. Hạn mức khách sạn là 1.200.000đ/đêm trong nước.",
    },
    {
        "id": "HR-011", "category": "Đào tạo",
        "title": "Ngân sách đào tạo",
        "text": "Mỗi nhân viên có ngân sách đào tạo 5.000.000đ/năm cho khóa học liên quan công việc. Cần đề xuất và được quản lý duyệt trước. Chứng chỉ đạt được phải nộp cho HR để lưu hồ sơ.",
    },
    {
        "id": "HR-012", "category": "Quy tắc",
        "title": "Quy định trang phục",
        "text": "Trang phục công sở lịch sự (smart casual) vào các ngày trong tuần. Thứ Sáu được mặc tự do (casual Friday) nhưng vẫn phải gọn gàng. Khi gặp khách hàng bắt buộc mặc trang phục công sở trang trọng.",
    },
    {
        "id": "HR-013", "category": "Quy tắc",
        "title": "Quy tắc ứng xử & xung đột lợi ích",
        "text": "Nhân viên không được nhận quà có giá trị trên 500.000đ từ đối tác. Mọi quan hệ kinh doanh với người thân phải khai báo với bộ phận Tuân thủ (Compliance). Vi phạm quy tắc ứng xử có thể dẫn đến chấm dứt hợp đồng.",
    },
    {
        "id": "HR-014", "category": "Nghỉ phép",
        "title": "Nghỉ thai sản",
        "text": "Lao động nữ được nghỉ thai sản 6 tháng theo luật, hưởng chế độ bảo hiểm xã hội. Lao động nam được nghỉ 5 ngày làm việc khi vợ sinh thường, 7 ngày khi sinh mổ, trong vòng 30 ngày đầu sau sinh.",
    },
    {
        "id": "HR-015", "category": "Nghỉ việc",
        "title": "Quy trình thôi việc",
        "text": "Nhân viên muốn nghỉ việc phải báo trước 30 ngày (nhân viên) hoặc 45 ngày (quản lý) bằng đơn chính thức. Cần bàn giao công việc và tài sản công ty trước ngày làm việc cuối. Lương và sổ BHXH được chốt trong 14 ngày sau khi nghỉ.",
    },
    # ---------------- Văn phòng / Vận hành ----------------
    {
        "id": "OPS-001", "category": "Văn phòng",
        "title": "Đặt phòng họp",
        "text": "Đặt phòng họp qua Outlook Calendar hoặc app MeetingRoom. Phòng họp lớn (trên 10 người) cần đặt trước ít nhất 1 ngày. Nếu không sử dụng, hãy hủy đặt phòng để người khác dùng.",
    },
    {
        "id": "OPS-002", "category": "Văn phòng",
        "title": "Thẻ ra vào & khách đến",
        "text": "Thẻ ra vào là vật dụng cá nhân, không cho mượn. Mất thẻ báo ngay Lễ tân để khóa thẻ, phí cấp lại là 100.000đ. Khách đến làm việc phải được đăng ký trước và có nhân viên đón tại sảnh.",
    },
    {
        "id": "OPS-003", "category": "An toàn",
        "title": "Quy trình thoát hiểm khi có cháy",
        "text": "Khi có chuông báo cháy, giữ bình tĩnh, không dùng thang máy, theo lối thoát hiểm gần nhất đến điểm tập kết tại bãi xe tầng trệt. Đội ứng cứu (mặc áo vàng) sẽ hướng dẫn. Diễn tập PCCC tổ chức 6 tháng một lần.",
    },
    {
        "id": "OPS-004", "category": "IT Helpdesk",
        "title": "Mức độ ưu tiên ticket (SLA)",
        "text": "Ticket IT có 4 mức ưu tiên: P1-Khẩn cấp (xử lý trong 1 giờ), P2-Cao (4 giờ), P3-Trung bình (1 ngày làm việc), P4-Thấp (3 ngày làm việc). Sự cố toàn hệ thống luôn là P1.",
    },
]


def get_corpus() -> List[Dict]:
    return CORPUS


def write_corpus_jsonl(path: str = "data/corpus.jsonl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in CORPUS:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"✅ Đã ghi {len(CORPUS)} chunks vào {path}")


if __name__ == "__main__":
    write_corpus_jsonl()

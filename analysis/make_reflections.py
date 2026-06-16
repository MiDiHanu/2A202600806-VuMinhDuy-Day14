"""
Sinh file reflection cá nhân cho từng thành viên từ template.

Chạy:
    python analysis/make_reflections.py "Nguyen Van A" "Tran Thi B" ...
Mỗi tên tạo ra analysis/reflections/reflection_<Ten>.md (không ghi đè nếu đã tồn tại).
"""
import io
import os
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "reflections", "reflection_TEMPLATE.md")


def slug(name: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn")
    s = s.replace("đ", "d").replace("Đ", "D")
    return "_".join(s.split())


def main():
    names = sys.argv[1:]
    if not names:
        print("Cách dùng: python analysis/make_reflections.py \"Tên SV 1\" \"Tên SV 2\" ...")
        return
    with open(TEMPLATE, encoding="utf-8") as f:
        tmpl = f.read()
    for name in names:
        path = os.path.join(HERE, "reflections", f"reflection_{slug(name)}.md")
        if os.path.exists(path):
            print(f"⏭️  Bỏ qua (đã tồn tại): {path}")
            continue
        content = tmpl.replace("_[Điền tên]_", name, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Tạo {path}")


if __name__ == "__main__":
    main()

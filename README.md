# AI Movie Short & Review Studio

Ứng dụng desktop chạy local trên Windows, hỗ trợ tạo **một video review hoàn chỉnh** từ video nguồn, lời thoại do người dùng nhập, giọng đọc, phụ đề và các tùy chỉnh xuất bản.

Luồng chính được tổ chức theo 4 màn hình tuần tự:

```text
Video & Project → Lời thoại & Ghép giọng → Phụ đề → Xuất bản
```

---

## Yêu cầu hệ thống

- **Windows 10/11**
- **Python 3.10+** → [Tải tại đây](https://www.python.org/downloads/)
- Kết nối internet lần đầu để tải FFmpeg và yt-dlp tự động
- Kết nối internet khi tải video hoặc dùng dịch vụ AI/TTS trực tuyến

### Tùy chọn: VoxCPM2 local

VoxCPM không nằm trong bộ phụ thuộc mặc định vì PyTorch và model chiếm vài GB.
Muốn dùng tạo giọng local/clone giọng:

```text
Double-click file install_voxcpm.bat
```

Sau khi cài xong, mở lại app và chọn **Bước 2 → Provider → VoxCPM2**.
Model `openbmb/VoxCPM2` được tải tự động ở lần tạo giọng đầu tiên.
Bạn cũng có thể dùng `voxcpm.bat` để kiểm tra runtime và luôn khởi động app
bằng đúng Python trong `venv`.

- Python khuyến nghị: 3.10–3.12.
- NVIDIA GPU: CUDA 12+, khoảng 8 GB VRAM cho VoxCPM2.
- GPU dưới mức này: chọn **Tự động** để app chuyển sang CPU; vẫn chạy được
  nhưng thời gian tạo giọng sẽ lâu hơn đáng kể.
- Audio clone nên sạch, dài khoảng 5–30 giây.

---

## Cách chạy

**Cách 1 – Nhanh nhất:**

```text
Double-click file run.bat
```

**Cách 2 – Thủ công:**

```bash
# Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate

# Cài dependencies
pip install -r requirements.txt

# Chạy app
python main.py
```

---

## Lần đầu khởi động

1. Ứng dụng sẽ hỏi tải FFmpeg (khoảng 80 MB) → Chọn **Yes**.
2. Vào **⚙ Cài đặt → Dependencies** → Nhấn **Tải tất cả**.
3. Chờ tải xong; FFmpeg và yt-dlp sẽ được lưu trong `tools/`.

---

## Quy trình 4 bước

### 1. Video & Project

- Kéo thả hoặc chọn một file video trên máy.
- Có thể dán URL YouTube, TikTok, Facebook/Reels và Douyin để tải video.
- Video Facebook không công khai có thể cần bật cookies của trình duyệt đang
  đăng nhập; chỉ tải nội dung bạn có quyền sử dụng.
- Có thể chọn **Facebook Reels** tại mục **Toàn bộ video** và tải theo từng
  đợt (10/20/50...). Tool mở một hồ sơ Chrome/Edge riêng để đăng nhập, tự cuộn
  tab Reels và lưu lịch sử; lần chạy sau sẽ bỏ qua Reel đã tải.
- Hỗ trợ các định dạng phổ biến như `mp4`, `mkv`, `mov`, `avi`, `webm`.
- Tạo project và kiểm tra đúng video trước khi sang bước tiếp theo.

Video nguồn là **bản gốc bất biến**. Ứng dụng chỉ đọc video này và tạo các sản phẩm trung gian riêng; không ghi đè nội dung, không chèn chữ `PART`, giọng đọc hoặc phụ đề vào bản nguồn.

### 2. Lời thoại & Ghép giọng

- Dán toàn bộ lời thoại vào ô nội dung, không cần phiên âm nguồn, chia cảnh hoặc qua màn kịch bản riêng.
- Chọn nhà cung cấp, giọng và tốc độ đọc theo hệ số như `0.90x`, `1.00x`, `1.10x`.
- Chọn **Ngọc Huyền (mới) — local** để dùng model NGHI-TTS. Model được tải một lần
  rồi tạo giọng trực tiếp trên máy.
- Provider NGHI-TTS có toàn bộ danh sách giọng từ `nghitts.app/api/models`;
  mỗi model chỉ được tải khi chọn dùng lần đầu.
- Bấm **Nghe thử nhanh** để phát một câu mẫu ngắn; app lưu mẫu theo giọng, tốc độ
  và âm lượng nên lần nghe lại không phải tạo lại.
- Chọn **Lồng tiếng theo mốc [giây] — tự dịch**, rồi bấm **Quét lời Trung theo
  mốc thời gian**. App đưa lời nguồn vào ô sửa theo dạng `[0.0s] ...`, `[13.4s] ...`.
  Bạn tự dịch hoặc sửa nội dung nhưng giữ nguyên các mốc, sau đó bấm **Tạo voice &
  ghép theo các mốc**. Chế độ này không gọi AI dịch. Toàn bộ câu dùng một nhịp
  đọc chung như một phiên dịch viên: câu bắt đầu ở đúng mốc nếu câu trước đã xong;
  nếu chưa xong thì câu sau nối ngay sau câu trước. Các mốc nhỏ được gom thành
  cụm thường dài 4–8 giây, ưu tiên kết thúc câu và không vượt khoảng nghỉ dài.
  Kiểm tra trước khi tạo voice và gợi ý số từ đều tính theo cùng các cụm này.
  TTS đọc liền cụm, sau đó bỏ im lặng thừa ở hai đầu, giữ khoảng nghỉ bên trong.
  Script nhập vẫn giữ nguyên các mốc; timing cụm thực tế lưu trong `voice_groups`.
  Trong chế độ lồng tiếng theo
  mốc, tool đo audio thực tế và chọn một nhịp chung từ **1.10x đến 1.15x**.
  Nếu vẫn vượt video ở 1.15x, tool báo số giây dư để bạn rút gọn lời;
  không tăng vượt giới hạn hoặc cắt mất lời đọc.
- Với VoxCPM2 có ba chế độ: Voice Design, clone bằng audio mẫu và Hi-Fi clone
  bằng audio mẫu kèm transcript chính xác.
- Bấm **Tạo giọng đọc**, nghe thử rồi bấm **Ghép giọng vào video**.
- Chọn **Tắt hoàn toàn âm thanh gốc của video** hoặc giữ một phần âm gốc bằng thanh âm lượng.
- Ghép xong, ứng dụng tự chuyển sang màn hình Phụ đề.

Kết quả của bước này là một bản review có giọng đọc nhưng **chưa có phụ đề và không có chữ `PART`**. Không ghép giọng vào các video Part đã xuất hoặc video đã burn chữ trước đó.

### 3. Phụ đề

- Chỉ tạo phụ đề từ **giọng narration/review đã duyệt** hoặc từ timing của chính các câu narration.
- Không dùng transcript lời thoại của video nguồn làm phụ đề cuối.
- Kiểm tra lại thời gian, nội dung và kiểu hiển thị sau khi tốc độ giọng đọc đã được khóa.

Nhờ vậy phụ đề bám đúng câu review đang phát. Nếu thay giọng hoặc tốc độ đọc, hãy tạo/căn lại phụ đề trước khi xuất.

### 4. Xuất bản

- Có thể bật **Nhạc nền**, chọn file MP3/WAV/M4A/AAC/FLAC/OGG và chỉnh âm lượng ngay tại tab xuất bản. Nhạc được tự động lặp đến hết video; mức 8–20% thường đủ nghe mà không lấn giọng đọc.

- Xem lại bản review có giọng, phụ đề và các tùy chọn trình bày cuối.
- Chọn cách hiển thị phụ đề rồi xuất **một video review hoàn chỉnh**.
- Mỗi lần xuất đều dựng lại từ các artifact sạch của project, không nối tiếp từ file Part đã có chữ.
- Để che phụ đề nước ngoài đã dính vào hình, bật **Xử lý watermark**, chọn
  **Vá nền mềm (Delogo)**, rồi kéo vùng chọn ôm sát dòng chữ cũ. Vùng chọn có thể
  phủ hết chiều ngang video khi phụ đề cũ chạy dài toàn khung. Phụ đề tiếng Việt
  được vẽ sau bước vá nền nên luôn nằm phía trên.

Luồng review tiêu chuẩn kết thúc ở một file xuất cuối và không cần chạy thêm công cụ Ghép video.

---

## Nguồn bất biến và artifact tách riêng

Mỗi công đoạn tạo ra một artifact độc lập:

| Công đoạn | Artifact chính | Mục đích |
|---|---|---|
| Nguồn | Video nguồn gốc | Mốc hình ảnh và âm thanh không bị thay đổi |
| Lời thoại & Ghép giọng | Audio narration và review có voice | Tạo giọng rồi ghép với video nguồn |
| Phụ đề lời đọc | Narration transcript/subtitle | Phụ đề chỉ cho lời review |
| Xuất bản | Final review | File bàn giao cuối cùng |

Khi thay đổi một bước, các artifact phía sau có thể hết hiệu lực và cần được tạo lại. Cách làm này tránh việc chữ `PART`, phụ đề cũ hoặc audio cũ bị “dính” vĩnh viễn vào lần xuất mới.

> Chữ đã được burn vào một video xuất cũ không thể gỡ sạch bằng thao tác ghép audio. Muốn có bản không chữ, phải dựng lại từ video nguồn hoặc video nền sạch.

---

## Cấu trúc thư mục

```text
TOOL/
├── main.py                 # Entry point
├── run.bat                 # Launcher
├── requirements.txt
├── tools/                  # FFmpeg, yt-dlp
├── data/
│   └── api_keys.json       # API keys lưu local
├── output/
│   └── <ProjectName>/
│       ├── project.json    # Trạng thái và đường dẫn artifact
│       ├── clips.json      # Kế hoạch phân cảnh
│       ├── audio/          # Transcript và audio narration
│       ├── subtitles/      # Phụ đề narration
│       ├── previews/       # File xem thử
│       ├── final/          # Bản nền sạch/bản review có voice
│       ├── exports/        # Video review đã xuất
│       └── export_history.json
├── logs/
│   └── app.log
└── src/
    ├── core/               # Logic nghiệp vụ
    ├── models/             # Data models
    ├── ui/                 # Giao diện PyQt6
    └── utils/              # Tiện ích
```

Tên file cụ thể có thể thay đổi theo project; đường dẫn artifact hiện hành được lưu trong `project.json`.

---

## Tính năng chính

- Nhập video local hoặc tải video từ URL.
- Phiên âm nguồn để phân tích nội dung.
- Đề xuất và chỉnh sửa phân cảnh bằng AI hoặc thủ công.
- Viết kịch bản review theo từng cảnh.
- Tạo TTS theo cảnh bằng edge-tts, NGHI-TTS/Piper hoặc OmniVoice Studio.
- Lồng tiếng Trung sang Việt theo timestamp bằng mọi giọng Edge, NGHI-TTS hoặc VoxCPM.
- Tạo giọng local bằng VoxCPM2: voice design, clone giọng và Hi-Fi clone.
- Căn tốc độ đọc theo hệ số `x` và ghép voice vào video nền sạch.
- Tạo phụ đề narration-only sau khi giọng đọc đã hoàn tất.
- Xuất một video review hoàn chỉnh.
- Tiện ích Tìm kênh và Ghép video hoạt động độc lập trên toolbar.

---

## Cài đặt API Keys

Vào **⚙ Cài đặt → API Keys** và nhập khóa của nhà cung cấp bạn sử dụng, chẳng hạn:

- Gemini
- Groq
- OpenRouter

Ollama chạy local nên không cần API key. TTS hiện hỗ trợ edge-tts,
NGHI-TTS/Piper, VoxCPM2 local và kết nối OmniVoice Studio tùy cấu hình.

Keys được lưu local tại `data/api_keys.json` và không được ứng dụng tự động tải lên nơi khác.

---

## Ghi chú kỹ thuật

- FFmpeg xử lý video trên máy.
- Cần internet khi tải video và khi dùng nhà cung cấp AI/TTS trực tuyến; các bước dựng bằng FFmpeg chạy local.
- Log chi tiết nằm tại `logs/app.log`.
- Ứng dụng hỗ trợ đường dẫn tiếng Việt; một số tài nguyên có thể được sao chép tạm sang đường dẫn ASCII để FFmpeg xử lý ổn định.

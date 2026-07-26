# VideoLingo — Hướng dẫn chạy (cấu hình riêng cho máy này)

> File này viết riêng cho máy `MS-7D99` sau khi cài xong ngày 2026-07-25.
> Toàn bộ môi trường đã được cài và **kiểm tra chạy thật** (CUDA + Streamlit đều OK).

---

## 1. Phần cứng máy bạn

| Thành phần | Thông số | Ghi chú |
|---|---|---|
| CPU | Intel i5-14400F — 10 nhân / 16 luồng | Đủ dùng |
| RAM | 80 GB | Rất dư |
| GPU | **2× RTX 5060 Ti 16GB** (Blackwell, compute capability **12.0 / sm_120**) | Chạy WhisperX `large-v3` ở float16 thoải mái |
| Driver | 610.74 (CUDA 13.3) | Mới, hỗ trợ đầy đủ |
| Ổ đĩa | C: còn 35 GB · **E: còn 343 GB** · F: còn 2.6 TB | C: khá chật → xem mục 7 |
| ffmpeg | `C:\tools\ffmpeg\bin` — có **NVENC** (`h264_nvenc`, `hevc_nvenc`) | Đã có sẵn trong PATH |
| Python hệ thống | 3.12 + 3.11 (dự án cần 3.10) | `uv` đã tự tải Python 3.10 riêng |

---

## 2. Đã cài những gì, nằm ở đâu

- **Môi trường ảo:** `E:\repos\VideoLingo\.venv` (Python 3.10.20) — đặt trên ổ E vì ổ C sắp đầy.
- **uv:** `C:\Users\ADMIN\.local\bin\uv.exe`
- **Các gói chính đã kiểm tra:**

  | Gói | Phiên bản |
  |---|---|
  | torch / torchaudio | **2.8.0+cu128** |
  | ctranslate2 | 4.8.1 (thấy đủ 2 GPU) |
  | whisperx | 3.8.6 |
  | spacy | 3.8.14 |
  | demucs | 4.1.0a3 |
  | streamlit | 1.49.1 |

### ⚠️ Điểm quan trọng nhất: vì sao phải cài PyTorch thủ công

`installer.py` dò phiên bản CUDA bằng regex `CUDA Version:` ([installer.py:171](installer.py#L171)).
Driver mới của bạn in ra `CUDA **UMD** Version: 13.3` → regex **không khớp** → script rơi về nhánh mặc định
`cu126` ([installer.py:191](installer.py#L191)).

Bản `cu126` **không chứa kernel cho sm_120**, GPU Blackwell của bạn sẽ báo lỗi:

```
CUDA error: no kernel image is available for execution on the device
```

Nên PyTorch đã được cài tay bằng `cu128` **trước**, rồi mới chạy installer (installer thấy torch 2.8.0
đã có sẵn nên bỏ qua bước cài torch). Kết quả kiểm tra:

```
arch list: ['sm_61','sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']  ← có sm_120 ✅
cuda available: True · device count: 2 · bf16: True
```

👉 **Đừng bao giờ chạy `installer.py --force`** — nó sẽ cài đè torch bằng `cu126` và làm hỏng GPU.
Nếu lỡ chạy, xem mục 8 để sửa.

---

## 3. Bắt buộc làm trước khi chạy lần đầu — điền API key

VideoLingo **phải có một LLM API** để dịch (đây là phần chính, không chạy local được).
Mở `config.yaml`, sửa khối này:

```yaml
api:
  key: 'sk-...'                      # ← điền key thật của bạn
  base_url: 'https://yunwu.ai'       # ← đổi theo nhà cung cấp bạn dùng
  model: 'gpt-5.5'                   # ← đổi theo model bạn có
  llm_support_json: false
```

Ví dụ nếu dùng thẳng API của Anthropic/OpenAI thì đổi `base_url` và `model` tương ứng.
Bạn cũng có thể điền trực tiếp trong sidebar của giao diện Streamlit sau khi mở app —
app sẽ ghi ngược lại vào `config.yaml`.

Đồng thời chỉnh dòng này cho hợp với bạn:

```yaml
display_language: "en"   # giao diện: en / zh-CN / zh-HK / ja / es / ru / fr (chưa có tiếng Việt)
```

---

## 3b. Dịch ra tiếng Việt — được, đã kiểm tra kỹ

`target_language` **đã được đặt sẵn thành `'Vietnamese'`** trong `config.yaml`.

### Vì sao chắc chắn chạy được

`target_language` là **chữ tự do**, chỉ được nhét thẳng vào prompt gửi cho LLM
([prompts.py:55](core/prompts.py#L55), [:145](core/prompts.py#L145), [:191](core/prompts.py#L191), [:253](core/prompts.py#L253)).
Không có danh sách ngôn ngữ hợp lệ nào để kiểm tra, nên viết gì cũng được —
`'Vietnamese'` hoặc `'Tiếng Việt'` đều chạy. Dùng `'Vietnamese'` thì an toàn hơn chút vì
toàn bộ prompt còn lại viết bằng tiếng Anh.

Chỗ **duy nhất** trong code có whitelist ngôn ngữ và ném lỗi là
`get_joiner()` ([config_utils.py:50-55](core/utils/config_utils.py#L50-L55)) — nó raise
`ValueError: Unsupported language code`. Tôi đã kiểm tra cả 5 nơi gọi hàm này
(`split_by_mark.py`, `split_long_by_root.py`, `_3_2_split_meaning.py`, `_5_split_sub.py`):
**tất cả đều truyền vào ngôn ngữ NGUỒN (từ whisper), không bao giờ truyền ngôn ngữ đích.**
Nên tiếng Việt ở vế đích không đụng tới nó.

Font phụ đề trên Windows là `Arial` ([_7_sub_into_vid.py:10](core/_7_sub_into_vid.py#L10)) —
hiển thị đầy đủ dấu tiếng Việt, không cần cài thêm gì.

### ⚠️ Giới hạn: video NÓI tiếng Việt thì KHÔNG chạy được

Đây là chiều ngược lại và nó **không** hoạt động. `whisper.language` (ngôn ngữ nguồn) chỉ nhận
8 mã có model spaCy: `en, ru, fr, ja, es, de, it, zh`. Nếu đặt `language: 'vi'`:

- không có model spaCy cho tiếng Việt trong `spacy_model_map`
- `get_joiner('vi')` sẽ ném `ValueError: Unsupported language code: vi`

Tóm lại: **video tiếng Anh/Trung/Nhật... → phụ đề tiếng Việt: OK.
Video tiếng Việt → ngôn ngữ khác: không hỗ trợ.**

### Lồng tiếng Việt (dubbing)

Dùng **edge-tts** — miễn phí, không cần API key. Đã test thật trên máy bạn, ra file audio 4.15 giây OK.

```yaml
tts_method: 'edge_tts'

edge_tts:
  voice: 'vi-VN-HoaiMyNeural'   # giọng nữ
  # voice: 'vi-VN-NamMinhNeural' # giọng nam
```

Đó là 2 giọng tiếng Việt duy nhất mà edge-tts có (đã liệt kê từ máy bạn).
Các backend khác (`azure_tts`, `openai_tts`, `fish_tts`, `f5tts`) đều cần key 302.ai;
`gpt_sovits` không có model tiếng Việt sẵn.

### Lưu ý về độ dài phụ đề

Tiếng Việt dài hơn tiếng Anh khoảng 10–20%. `target_multiplier: 1.2` trong config đã tính sẵn
phần này. Nếu thấy dòng phụ đề bị tràn màn hình, hạ `subtitle.max_length` từ 75 xuống ~60.

---

## 4. Cách chạy

### Cách 1 — dễ nhất

Nhấp đúp vào **`OneKeyStart.bat`** trong `E:\repos\VideoLingo\`.

Script tự tìm `.venv`, tự chạy kiểm tra môi trường, rồi mở app. Log lưu ở `logs\`.

### Cách 2 — chạy tay từ PowerShell

```powershell
cd E:\repos\VideoLingo
.\.venv\Scripts\python.exe -m streamlit run st.py
```

Sau đó mở trình duyệt vào **http://localhost:8501**

### Kiểm tra môi trường mà không mở app

```powershell
cd E:\repos\VideoLingo
.\.venv\Scripts\python.exe installer.py --check
```

Kết quả đúng (đã test) là in ra danh sách phiên bản, **không có dòng `ERROR:`** nào.

---

## 5. Quy trình dùng trên giao diện

1. **a. Tải video** — dán link YouTube, hoặc upload file (mp4/mov/avi/mkv/flv/wmv/webm,
   hoặc file audio wav/mp3/flac/m4a). Giới hạn upload đã đặt 4 GB.
2. **b. Translate and Generate Subtitles** — bấm `Start Processing Subtitles`. 5 bước:
   nhận dạng WhisperX → tách câu bằng NLP+LLM → tóm tắt & dịch nhiều bước →
   cắt/căn phụ đề → ghép phụ đề vào video.
3. **c. Dubbing** *(tuỳ chọn)* — bấm `Start Audio Processing` để lồng tiếng.
   Cần chọn `tts_method` và điền key TTS tương ứng trong `config.yaml`.

Có nút **⏸️ Pause / ⏹️ Stop** ngay trên thanh tiến trình nếu muốn dừng giữa chừng.

**Kết quả nằm ở:** `output\` (`output_sub.mp4`, `output_dub.mp4`, `src.srt`, `trans.srt`).
Bấm `Archive to 'history'` để dọn và lưu sang thư mục `history\` trước khi làm video tiếp theo.

**Lần chạy đầu sẽ chậm** vì phải tải model: WhisperX `large-v3` (~3 GB), model căn chỉnh,
model spaCy `en_core_web_md`. Từ lần 2 trở đi nhanh hơn nhiều.

---

## 6. Cấu hình khuyến nghị cho máy này

Sửa trong `config.yaml`:

```yaml
whisper:
  model: 'large-v3'     # giữ nguyên — 16GB VRAM chạy tốt, chất lượng cao nhất
  runtime: 'local'      # chạy trên GPU của bạn, không tốn tiền API

ffmpeg_gpu: true        # ⭐ BẬT LÊN — đã xác nhận ffmpeg của bạn có h264_nvenc,
                        #   ghép/nén video sẽ nhanh hơn nhiều so với CPU

demucs: true            # ⭐ nên bật nếu video có nhạc nền — tách giọng trước khi
                        #   nhận dạng, phụ đề chính xác hơn hẳn. Máy bạn thừa sức chạy.

max_workers: 8          # tăng từ 4 lên 8 để dịch nhanh hơn (i5-14400F có 16 luồng).
                        #   Nếu nhà cung cấp API báo lỗi rate limit thì hạ về 4.

ytb_resolution: '1080'  # giữ nguyên
```

Code tự chọn `batch_size = 16` và `compute_type = float16` khi thấy GPU > 8 GB
([whisperX_local.py:85-88](core/asr_backend/whisperX_local.py#L85-L88)) — đúng với máy bạn, không cần chỉnh gì.

---

## 7. Mẹo cho máy 2 GPU và chuyện ổ đĩa

### Chọn GPU

VideoLingo luôn dùng `cuda:0`. Lúc kiểm tra, **GPU 0 đang bị chiếm ~9 GB** (màn hình + app khác),
còn **GPU 1 gần như trống** (1.3 GB). Muốn ép dùng GPU 1 cho rảnh tay:

```powershell
cd E:\repos\VideoLingo
$env:CUDA_VISIBLE_DEVICES = "1"
.\.venv\Scripts\python.exe -m streamlit run st.py
```

Khi đặt biến này, GPU 1 sẽ trở thành `cuda:0` dưới góc nhìn của chương trình.
VideoLingo không chạy song song 2 GPU được, nên cách này chỉ để **chọn** card, không tăng tốc gấp đôi.

### Đẩy cache model sang ổ E (khuyến nghị mạnh)

Ổ C: chỉ còn **35 GB**. Mặc định model HuggingFace tải về `C:\Users\ADMIN\.cache\huggingface`
(khoảng 5+ GB). Đặt biến môi trường để chuyển sang ổ E:

```powershell
# chạy 1 lần, có hiệu lực vĩnh viễn cho user hiện tại
[Environment]::SetEnvironmentVariable("HF_HOME", "E:\hf_cache", "User")
```

Mở lại PowerShell/terminal sau khi chạy lệnh trên.

### File .bat tiện dụng (tuỳ chọn)

Nếu muốn gộp cả 2 mẹo trên, tạo file `Start_VL.bat` trong `E:\repos\VideoLingo\`:

```bat
@echo off
cd /D "%~dp0"
set "HF_HOME=E:\hf_cache"
set "CUDA_VISIBLE_DEVICES=1"
.venv\Scripts\python.exe -m streamlit run st.py
pause
```

---

## 8. Lỗi thường gặp và cách sửa

### `CUDA error: no kernel image is available for execution on the device`

Nghĩa là torch đã bị cài đè bằng bản `cu126` (thường do lỡ chạy `installer.py --force`).
Cài lại đúng bản:

```powershell
cd E:\repos\VideoLingo
.\.venv\Scripts\python.exe -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
```

Kiểm tra lại (phải thấy `sm_120` trong danh sách):

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_arch_list())"
```

### Muốn cài lại thư viện nhưng KHÔNG đụng tới torch

```powershell
.\.venv\Scripts\python.exe installer.py --yes
```

Lệnh này an toàn — nó thấy torch 2.8.0 đã có nên bỏ qua. Chỉ `--force` mới nguy hiểm.

### `ffmpeg not found in PATH`

Máy bạn đã có `C:\tools\ffmpeg\bin\ffmpeg.exe`. Nếu báo lỗi này thì PATH bị mất, thêm lại:

```powershell
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\tools\ffmpeg\bin", "User")
```

### Tải model bị đứt giữa chừng / kẹt ở `_model_cache`

Xoá thư mục cache rồi chạy lại — code sẽ tự chuyển sang cache toàn cục của HuggingFace:

```powershell
Remove-Item -Recurse -Force E:\repos\VideoLingo\_model_cache
```

### Hết VRAM (`out of memory`)

Đóng app đang chiếm GPU 0, hoặc dùng `CUDA_VISIBLE_DEVICES=1` (mục 7),
hoặc đổi `whisper.model` sang `large-v3-turbo` (nhẹ hơn, nhanh hơn, chính xác kém hơn chút).

### Port 8501 đã bị chiếm

```powershell
.\.venv\Scripts\python.exe -m streamlit run st.py --server.port 8502
```

---

## 9. Bảng lệnh nhanh

```powershell
cd E:\repos\VideoLingo

# chạy app
.\OneKeyStart.bat
# hoặc
.\.venv\Scripts\python.exe -m streamlit run st.py

# kiểm tra môi trường
.\.venv\Scripts\python.exe installer.py --check

# kiểm tra GPU
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"

# xử lý hàng loạt nhiều video
#   1) điền danh sách vào batch\tasks_setting.xlsx
#   2) chạy:
.\batch\OneKeyBatch.bat
```

---

## 10. Những gì đã được kiểm tra thật (không phải suy đoán)

- ✅ `torch 2.8.0+cu128`, `cuda.is_available() = True`, 2 GPU, `sm_120` có trong arch list
- ✅ Phép nhân ma trận trên GPU chạy được
- ✅ `ctranslate2 4.8.1` nhận đủ 2 GPU
- ✅ Chạy thật WhisperX (model `tiny`) trên GPU với `float16` → thành công (cuDNN 9 + cuBLAS 12 nạp OK)
- ✅ `installer.py --check` trả về exit code 0, không có ERROR
- ✅ Streamlit khởi động và trả về HTTP 200, không có lỗi stderr
- ✅ ffmpeg có `h264_nvenc` và `hevc_nvenc`

Chưa kiểm tra được (cần API key của bạn): bước **dịch bằng LLM** và bước **lồng tiếng TTS**.

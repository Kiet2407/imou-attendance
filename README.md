# Chấm công cửa ra vào bằng OpenCV

Bản thử nghiệm chạy cục bộ cho một camera: YuNet phát hiện mặt, SFace trích xuất đặc trưng và so sánh với nhân viên đã đăng ký; bộ theo dõi xác nhận di chuyển qua đường ranh theo chiều vào. SQLite lưu các lượt vào. Báo cáo CSV/HTML lấy lượt vào sớm nhất trong ngày theo `Asia/Ho_Chi_Minh` và hiển thị cả nhân viên chưa có ghi nhận.

## Chuẩn bị

Python 3.10 trở lên, camera có luồng video mà OpenCV đọc được, hoặc file video để thử. Kiểm tra model/firmware Imou trước để xác nhận cách lấy luồng RTSP; dự án không tự dò camera hoặc đọc trực tiếp tài khoản Imou Life.

```bash
git clone https://github.com/Kiet2407/imou-attendance.git
cd imou-attendance
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m attendance download-models
```

Môi trường `.venv`, model và dữ liệu cá nhân không được đưa vào repo. Chạy các bước trên sau khi clone. `requirements-tested.txt` ghi phiên bản đã chạy kiểm thử ở môi trường phát triển.

Các model lấy từ [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models), khóa theo revision trong `attendance/vision.py`. Xem giấy phép trong thư mục model tương ứng trước khi phân phối lại. Kiến trúc nhận diện theo [hướng dẫn chính thức OpenCV](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html).

## 1. Đăng ký nhân viên

Có thể lấy trực tiếp từ webcam, không cần chuẩn bị file ảnh:

```bash
.venv/bin/python -m attendance enroll-camera --id NV001 --name 'Dakin' --source 0
```

Nhìn vào webcam, chỉ một người trong hình. Lệnh chờ 2 giây rồi tự lấy 5 mẫu mặt cách nhau ít nhất 0,8 giây trong tối đa 30 giây. Mẫu sau phải khớp với mẫu đầu. Terminal báo tiến độ, không mở cửa sổ GUI và không lưu ảnh gốc. Chỉ lưu vector sau khi lấy đủ mẫu; đây là kiểm tra nhất quán, chưa xác thực danh tính hoặc liveness. Nếu không mở được thiết bị, kiểm tra webcam/quyền truy cập ở máy chạy Python; môi trường container/remote cần được cấp thiết bị riêng.

Hoặc đăng ký từ ảnh có sẵn:

Dùng 3–5 ảnh chụp thật, rõ mặt, có góc nhìn và ánh sáng gần giống camera cửa. Mỗi ảnh chỉ chứa một khuôn mặt. Không dùng ảnh AI sinh làm mẫu nhận diện danh tính.

```bash
.venv/bin/python -m attendance enroll --id NV001 --name 'Nguyễn Văn An' --images /duong-dan/an-1.jpg /duong-dan/an-2.jpg
```

Lệnh kiểm tra toàn bộ ảnh trước khi lưu; mỗi lần đăng ký tiếp sẽ bổ sung mẫu. Chỉ ảnh đọc được và có đúng một mặt đủ lớn mới được chấp nhận. Ảnh gốc không được sao chép vào database; database lưu vector khuôn mặt. Nên đăng ký trước khi bắt đầu tiến trình camera; khi thêm mẫu, khởi động lại tiến trình để nạp danh sách mới.

Có thể thêm người vào danh sách trước khi có ảnh, để báo cáo vẫn hiển thị trạng thái chưa ghi nhận:

```bash
.venv/bin/python -m attendance employee --id NV002 --name 'Trần Thị Bình'
```

## 2. Chạy camera và chỉnh chiều vào

Nguồn webcam thử nghiệm:

```bash
.venv/bin/python -m attendance run --source 0 --axis x --line 0.5 --direction 1 --preview data/preview.jpg
```

Với Imou, đặt URL luồng đã xác minh trong biến môi trường `CAMERA_SOURCE`, rồi chạy:

```bash
.venv/bin/python -m attendance run --axis x --line 0.5 --direction 1 --preview data/preview.jpg
```

Không đưa URL chứa mật khẩu vào kho mã. OpenCV/FFmpeg có thể in URL trong thông báo lỗi của thư viện; tránh chia sẻ log thô chứa thông tin đăng nhập.

- `--axis x`: đường ranh thẳng đứng; `--direction 1`: từ trái qua phải là vào, `-1`: từ phải qua trái.
- `--axis y`: đường ranh ngang; `--direction 1`: từ trên xuống là vào, `-1`: từ dưới lên.
- `--line 0.5`: đường ranh ở 50% chiều rộng/cao của ảnh. Vùng đệm mặc định mỗi bên 4% để tránh rung quanh đường ranh.
- `--preview`: ghi đè ảnh JPEG có đường ranh và tâm mặt định kỳ; mở file để chỉnh vị trí. Bản headless không mở cửa sổ GUI.
- `--confirmations 3`: cần ít nhất ba lần nhận diện liên tiếp cùng danh tính trên một track trước hoặc tại thời điểm qua đường ranh. Khung hình không xác định danh tính sẽ xóa bộ đếm xác nhận.
- `--threshold 0.45`, `--margin 0.05`: ngưỡng cosine và chênh lệch với người giống thứ hai. Đây là điểm khởi đầu cần hiệu chỉnh bằng dữ liệu thực tế, không phải xác suất đúng.

Cần đặt đường ranh tại vị trí thấy mặt ở cả hai phía. Chỉ nhìn rõ mặt sau khi đã qua ranh có thể gây bỏ sót. Khi mất liên kết theo dõi hoặc nhiều người quá sát nhau, hệ thống ưu tiên bỏ lượt không chắc chắn. Lượt đi ra không được lưu thành giờ đến. Chỉ khởi chạy một tiến trình cho mỗi nguồn camera.

Thử với video có sẵn phải cung cấp **thời điểm bắt đầu quay thực tế**, có múi giờ:

```bash
.venv/bin/python -m attendance --db data/test.sqlite3 run --source /duong-dan/cua.avi --video-start '2026-11-25T07:30:00+07:00' --axis x --line 0.5 --direction 1
```

Thời gian file được tính bằng chỉ số khung hình/FPS; bản đầu giả định video FPS cố định. Video tua nhanh, bị cắt nối hoặc FPS biến đổi cần được xử lý thời gian riêng. Nên dùng database thử riêng để không trộn với chấm công thật. Luồng trực tiếp có luồng đọc riêng, chỉ giữ khung hình mới nhất và dùng thời điểm máy nhận khung hình; không xếp hàng toàn bộ khung hình trong lúc nhận diện. Độ trễ từ camera/mạng vẫn có thể tồn tại. Đồng bộ giờ máy trước khi sử dụng. Khi mất hình, chương trình thử kết nối lại và xóa trạng thái track cũ, không suy đoán giờ trong khoảng mất hình.

## 3. Xem báo cáo cuối ngày

```bash
.venv/bin/python -m attendance report --date 2026-11-25 --format html --output reports/2026-11-25.html
.venv/bin/python -m attendance report --date 2026-11-25 --format csv --output reports/2026-11-25.csv
```

Mở HTML bằng trình duyệt hoặc CSV bằng Excel. Bảng gồm mã nhân viên, tên, giờ đến đầu tiên, camera và trạng thái. Không có lượt nhận diện sẽ hiện **Chưa ghi nhận**, không tự kết luận vắng. Không truyền `--date` sẽ dùng ngày hiện tại tại Việt Nam. Có thể chọn database bằng `--db` đặt trước tên lệnh.

Báo cáo được tạo khi chạy lệnh; chưa có lịch xuất tự động hoặc máy chủ web. Dữ liệu vào được ghi tự động trong khi tiến trình camera hoạt động. Muốn sử dụng thực tế liên tục cần cấu hình tiến trình tự khởi động và lịch xuất báo cáo theo môi trường triển khai.

## Kiểm thử và giới hạn

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Đã kiểm tra: chiều vào/ra, không ghi người xuất hiện sẵn phía trong, từ chối track đổi danh tính/mơ hồ, ngày theo múi giờ, nhiều lượt lấy giờ sớm nhất, nhân viên chưa có lượt, xuất báo cáo và đọc video thật qua OpenCV với phần nhận diện giả lập. Hai model thật đã nạp và chạy suy luận trên ảnh trống để kiểm tra tương thích; điều này không đánh giá độ chính xác khuôn mặt.

Chưa thử trên camera Imou, ảnh nhân viên hoặc video văn phòng vì chưa có dữ liệu/kết nối. Chưa có phát hiện ảnh giả/màn hình (liveness), giao diện sửa chấm công, ảnh bằng chứng theo sự kiện hoặc hàng đợi kiểm duyệt người chưa nhận diện. Bộ theo dõi tâm mặt phù hợp thử nghiệm cửa ít người, chưa thay thế bộ theo dõi đa đối tượng cho cửa đông người. Chỉ xử lý mặt từ 70px mỗi chiều; độ mờ, góc nghiêng và ánh sáng vẫn ảnh hưởng kết quả. Không nên dùng kết quả chưa đối chiếu để tự động tính lương.

Dữ liệu vector khuôn mặt vẫn là dữ liệu nhạy cảm: thống nhất cách sử dụng với nhân viên, giới hạn người truy cập thư mục dữ liệu và quy định thời gian lưu. Ảnh preview nếu bật sẽ chứa hình camera; tắt tùy chọn này khi không cần hiệu chỉnh. Chương trình không gọi dịch vụ nhận diện đám mây; tải model là bước tải mạng riêng.

# === IMPORTS ===
import sys
import cv2
import serial
import serial.tools.list_ports
import time
import os
from datetime import datetime
import numpy as np # Cần cho audio và cv2

# --- Thư viện Audio ---
# <<< THÊM MỚI: Import thư viện audio >>>
try:
    import sounddevice as sd
    import soundfile as sf
    # Kiểm tra nhanh xem có thiết bị đầu vào nào không
    if not any(d['max_input_channels'] > 0 for d in sd.query_devices()):
        print("CẢNH BÁO: Không tìm thấy thiết bị ghi âm (microphone) nào.")
except ImportError:
    print("\n=====================================================")
    print(" LỖI: Vui lòng cài đặt thư viện 'sounddevice' và 'soundfile'. ")
    print(" Chạy lệnh sau trong terminal/command prompt:       ")
    print("   pip install sounddevice soundfile numpy             ")
    print("=====================================================\n")
    # Có thể hiện QMessageBox ở đây nếu muốn, nhưng import có thể thất bại trước khi app chạy
    # Thay vào đó, thoát chương trình để người dùng cài đặt
    sys.exit("Lỗi thiếu thư viện âm thanh. Vui lòng cài đặt và thử lại.")
except Exception as e:
    print(f"\nLỖI KHỞI TẠO ÂM THANH: {e}\n")
    # Không thoát ở đây, có thể vẫn dùng được video/serial
# --- /Thư viện Audio ---


from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QTextEdit,
                             QFileDialog, QGroupBox, QMessageBox, QSizePolicy,
                             QSpacerItem)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

# =============================================================================
# == Webcam Worker Thread (Giữ nguyên như code gốc) ==
# =============================================================================
class WebcamThread(QThread):
    """Handles video capture in a separate thread."""
    frame_ready = pyqtSignal(object)       # Emits the captured frame (numpy array)
    error = pyqtSignal(str)              # Emits error messages
    properties_ready = pyqtSignal(int, int, float) # Emits width, height, fps on successful open

    def __init__(self, webcam_index):
        super().__init__()
        self.webcam_index = webcam_index
        self.cap = None
        self._is_running = True
        self._width = 0
        self._height = 0
        self._fps = 0.0
        # print(f"Initializing WebcamThread for index {self.webcam_index}") # (Giữ log nếu muốn)

    def run(self):
        # print(f"WebcamThread {self.webcam_index}: Starting run loop.")
        self.cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            # print(f"Webcam {self.webcam_index}: Failed with MSMF, trying default backend...")
            self.cap = cv2.VideoCapture(self.webcam_index)
            if not self.cap.isOpened():
                self.error.emit(f"Không thể mở webcam {self.webcam_index} với bất kỳ backend nào.")
                self._is_running = False
                # print(f"WebcamThread {self.webcam_index}: Failed to open with any backend.")
                return

        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not (0 < self._fps < 150):
            # print(f"Warning: Invalid FPS ({self._fps:.2f}) detected for webcam {self.webcam_index}. Defaulting to 30.0")
            self._fps = 30.0
        self.properties_ready.emit(self._width, self._height, self._fps)
        # print(f"Webcam {self.webcam_index} opened successfully ({self._width}x{self._height} @ {self._fps:.2f} FPS)")

        while self._is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                if self._is_running:
                    self.error.emit(f"Mất kết nối với webcam {self.webcam_index} hoặc đọc frame thất bại.")
                    # print(f"WebcamThread {self.webcam_index}: Frame read failed or lost connection.")
                self._is_running = False
                break
            self.msleep(max(1, int(1000 / (self._fps * 1.5)))) # Điều chỉnh sleep dựa trên FPS

        if self.cap and self.cap.isOpened():
            # print(f"WebcamThread {self.webcam_index}: Releasing capture...")
            self.cap.release()
        # print(f"WebcamThread {self.webcam_index}: Exiting run loop.")

    def stop(self):
        """Requests the thread to stop."""
        # print(f"WebcamThread {self.webcam_index}: Stop requested.")
        self._is_running = False
        if not self.wait(2000):
             # print(f"Warning: Webcam thread {self.webcam_index} did not finish cleanly after 2s. Terminating.")
             self.terminate()
             if self.cap and self.cap.isOpened():
                  try:
                      self.cap.release()
                      # print(f"Webcam {self.webcam_index} capture released after termination.")
                  except Exception as e:
                      print(f"Error releasing webcam {self.webcam_index} after termination: {e}")
        # else:
            # print(f"WebcamThread {self.webcam_index}: Stopped successfully.")
        # Double-check release
        if self.cap and self.cap.isOpened():
             # print(f"Warning: Webcam {self.webcam_index} capture still open after wait(). Releasing fallback.")
             try: self.cap.release()
             except Exception as e: print(f"Error in fallback release for webcam {self.webcam_index}: {e}")


# =============================================================================
# == Audio Worker Thread ==
# =============================================================================
# <<< THÊM MỚI: Lớp AudioThread >>>
class AudioThread(QThread):
    """Handles audio recording in a separate thread using sounddevice and soundfile."""
    error = pyqtSignal(str)            # Emits error messages
    status_update = pyqtSignal(str)    # Emits status messages (e.g., started, stopped)
    finished_writing = pyqtSignal(str) # Emits the filename when writing is complete

    def __init__(self, filename, samplerate=44100, channels=1, device=None, blocksize=1024):
        """
        Initializes the AudioThread.

        Args:
            filename (str): Path to save the WAV file.
            samplerate (int): Sampling frequency in Hz.
            channels (int): Number of input channels (1 for mono, 2 for stereo).
            device (int or str, optional): Input device ID or substring. Defaults to None (system default).
            blocksize (int): The number of frames passed to the stream callback.
        """
        super().__init__()
        self.filename = filename
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self.blocksize = blocksize # Thêm blocksize để điều chỉnh
        self._is_running = True
        self._audio_file = None
        self._stream = None
        print(f"Initializing AudioThread: File='{os.path.basename(filename)}', Rate={samplerate}, Channels={channels}, Device={device}")

    def _audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(f"Audio Stream Status Warning: {status}", file=sys.stderr)
        # Ghi dữ liệu nhận được vào file WAV
        # Kiểm tra xem file còn mở và thread có đang chạy không
        if self._is_running and self._audio_file:
            try:
                self._audio_file.write(indata)
            except Exception as e:
                 # Lỗi này có thể xảy ra nếu file bị đóng bất ngờ
                 print(f"Error writing audio block: {e}", file=sys.stderr)
                 # Có thể emit lỗi hoặc cố gắng dừng thread ở đây
                 # self.error.emit(f"Lỗi ghi audio block: {e}")
                 # self._is_running = False # Dừng thread nếu ghi lỗi

    def run(self):
        """Starts the audio recording stream."""
        print(f"AudioThread ({os.path.basename(self.filename)}): Starting run loop.")
        self._is_running = True # Đảm bảo cờ được đặt khi bắt đầu

        try:
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)

            # Mở file WAV để ghi
            # subtype='PCM_16' là định dạng WAV phổ biến
            self._audio_file = sf.SoundFile(self.filename, mode='w', samplerate=self.samplerate,
                                            channels=self.channels, subtype='PCM_16')
            print(f"Audio file opened: {self.filename}")

            # Tạo và bắt đầu luồng ghi âm
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                device=self.device,
                channels=self.channels,
                callback=self._audio_callback,
                blocksize=self.blocksize # Sử dụng blocksize
            )
            self._stream.start()
            self.status_update.emit(f"Bắt đầu ghi âm thanh vào {os.path.basename(self.filename)}")

            # Giữ thread chạy miễn là _is_running là True
            # Luồng callback của sounddevice sẽ xử lý việc ghi dữ liệu
            while self._is_running:
                self.msleep(100) # Ngủ một chút để không chiếm CPU quá nhiều

            print(f"AudioThread ({os.path.basename(self.filename)}): Run loop requested to exit.")

        except sd.PortAudioError as pae:
             error_msg = f"Lỗi PortAudio ({self.device}): {pae}"
             print(error_msg, file=sys.stderr)
             self.error.emit(error_msg + "\nKiểm tra thiết bị âm thanh hoặc thử chọn thiết bị khác.")
             self._is_running = False
        except Exception as e:
            error_msg = f"Lỗi AudioThread không xác định: {e}"
            print(error_msg, file=sys.stderr)
            self.error.emit(error_msg)
            self._is_running = False # Dừng nếu có lỗi nghiêm trọng
        finally:
            print(f"AudioThread ({os.path.basename(self.filename)}): Entering finally block.")
            # --- Dọn dẹp tài nguyên ---
            stream_closed = False
            if self._stream:
                try:
                    if not self._stream.stopped: # Chỉ stop nếu chưa dừng
                         print("Stopping audio stream...")
                         self._stream.stop()
                    if not self._stream.closed: # Chỉ close nếu chưa đóng
                         print("Closing audio stream...")
                         self._stream.close()
                    stream_closed = True
                    print("Audio stream stopped and closed.")
                except sd.PortAudioError as pae_stop:
                     print(f"Error stopping/closing audio stream: {pae_stop}", file=sys.stderr)
                except Exception as e_stop:
                     print(f"Generic error stopping/closing audio stream: {e_stop}", file=sys.stderr)
                self._stream = None # Xóa tham chiếu

            file_closed = False
            if self._audio_file:
                try:
                    if not self._audio_file.closed:
                        print("Closing audio file...")
                        self._audio_file.close()
                        file_closed = True
                        print("Audio file closed.")
                except Exception as e_close:
                     print(f"Error closing audio file '{self.filename}': {e_close}", file=sys.stderr)
                self._audio_file = None # Xóa tham chiếu

            # Emit tín hiệu chỉ khi cả stream và file đã được đóng (hoặc không tồn tại)
            if stream_closed and file_closed:
                 self.finished_writing.emit(self.filename)
                 print(f"AudioThread confirmed finished writing: {os.path.basename(self.filename)}")
            else:
                 print(f"AudioThread finished writing confirmation SKIPPED (Stream closed: {stream_closed}, File closed: {file_closed})")

            print(f"AudioThread ({os.path.basename(self.filename)}): Exiting run loop.")

    def stop(self):
        """Requests the thread to stop recording."""
        print(f"AudioThread ({os.path.basename(self.filename)}): Stop requested.")
        self._is_running = False
        # Không cần gọi stream.stop() hay file.close() ở đây,
        # vì khối finally trong run() sẽ xử lý việc đó khi vòng lặp kết thúc.
        # Việc chờ thread kết thúc sẽ được thực hiện ở Main Window.


# =============================================================================
# == Serial Worker Thread (Giữ nguyên như code gốc) ==
# =============================================================================
class SerialThread(QThread):
    """Handles serial communication in a separate thread."""
    data_received = pyqtSignal(str) # Emits received lines
    error = pyqtSignal(str)         # Emits error messages

    def __init__(self, port, baudrate=9600, timeout=1):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_connection = None
        self._is_running = True
        # print(f"Initializing SerialThread: Port={self.port}, Baudrate={self.baudrate}")

    def run(self):
        # print(f"SerialThread {self.port}: Starting run loop.")
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            # print(f"Serial port {self.port} opened successfully at {self.baudrate} baud.")

            while self._is_running and self.serial_connection and self.serial_connection.isOpen():
                try:
                    if self.serial_connection.in_waiting > 0:
                        try:
                            line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                            if line:
                                self.data_received.emit(line)
                        except UnicodeDecodeError as ude:
                             print(f"Serial Decode Error on {self.port}: {ude}. Skipping.")
                except serial.SerialException as e:
                    if self._is_running:
                        self.error.emit(f"Lỗi đọc/ghi Serial ({self.port}): {e}")
                        # print(f"SerialThread {self.port}: SerialException: {e}")
                    self._is_running = False
                except OSError as e:
                     if self._is_running:
                        self.error.emit(f"Lỗi hệ thống cổng Serial ({self.port}): {e}")
                        # print(f"SerialThread {self.port}: OSError: {e}")
                     self._is_running = False
                except Exception as e:
                     if self._is_running:
                         self.error.emit(f"Lỗi Serial không xác định ({self.port}): {e}")
                         # print(f"SerialThread {self.port}: Unexpected Exception: {e}")
                     self._is_running = False

                self.msleep(50) # Yield CPU

        except serial.SerialException as e:
            self.error.emit(f"Không thể mở cổng Serial {self.port} tại {self.baudrate} baud: {e}")
            # print(f"SerialThread {self.port}: Failed to open port: {e}")
        except Exception as e:
             self.error.emit(f"Lỗi khởi tạo Serial không xác định ({self.port}): {e}")
             # print(f"SerialThread {self.port}: Failed to initialize: {e}")
             self._is_running = False

        finally:
            if self.serial_connection and self.serial_connection.isOpen():
                try:
                    # print(f"SerialThread {self.port}: Closing serial port in finally block...")
                    self.serial_connection.close()
                except Exception as e:
                     print(f"Error closing serial port {self.port} during run cleanup: {e}")
            # print(f"SerialThread ({self.port}) exiting run loop.")

    def stop(self):
        """Requests the thread to stop."""
        # print(f"SerialThread {self.port}: Stop requested.")
        self._is_running = False
        if self.serial_connection and self.serial_connection.isOpen():
            try:
                 # print(f"SerialThread {self.port}: Closing port from stop()...")
                 self.serial_connection.close()
            except Exception as e:
                print(f"Error closing serial port {self.port} in stop(): {e}")
        if not self.wait(1500):
            # print(f"Warning: Serial thread ({self.port}) did not finish cleanly after 1.5s. Terminating.")
            self.terminate()
        # else:
             # print(f"SerialThread ({self.port}) stopped successfully.")


# =============================================================================
# == Main Application Window ==
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Giám sát Webcam, Audio và Điều khiển Serial (v1.1)") # Cập nhật tiêu đề
        self.setGeometry(100, 100, 950, 800) # Tăng chiều cao một chút

        # --- State Variables ---
        self.webcam_thread = None
        self.serial_thread = None
        self.audio_thread = None # <<< THÊM MỚI: Biến cho audio thread
        self.video_writer = None
        self.is_recording = False
        self.is_paused = False # Pause hiện chỉ áp dụng cho video
        self.save_directory = os.getcwd()
        self.webcam_properties = {'width': None, 'height': None, 'fps': None}
        self.last_video_filename = ""
        self.last_audio_filename = "" # <<< THÊM MỚI: Tên file audio gần nhất

        self.recording_session_counter = 0

        # --- Audio Config (Có thể thêm UI để thay đổi sau) ---
        self.audio_samplerate = 44100
        self.audio_channels = 1 # Mono
        self.audio_device_index = None # None = Default device

        # --- Timers ---
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_status_visuals)
        self.recording_flash_state = False

        # --- Constants ---
        self.common_baud_rates = ["9600", "19200", "38400", "57600", "115200", "250000", "4800", "2400"]
        self.default_baud_rate = "9600"

        # --- Initialize UI ---
        self._init_ui()

        # --- Initial Scans & UI Updates ---
        self._scan_webcams()
        self._scan_serial_ports()
        self._scan_audio_devices() # <<< THÊM MỚI: Quét thiết bị audio
        self._update_save_dir_label()

        print("MainWindow initialized.")


    def _init_ui(self):
        """Build the user interface."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # --- 1. Video Display Area (Giữ nguyên) ---
        self.video_frame_label = QLabel("Chưa bật Webcam")
        # ... (Giữ nguyên cấu hình QLabel)
        self.video_frame_label.setMinimumSize(640, 480)
        self.main_layout.addWidget(self.video_frame_label, 1)

        # --- 2. Controls Area (Horizontal Layout) ---
        self.controls_area_widget = QWidget()
        self.controls_area_layout = QHBoxLayout(self.controls_area_widget)
        self.main_layout.addWidget(self.controls_area_widget)

        # --- 2a. Column 1: Webcam & Recording Controls ---
        col1_layout = QVBoxLayout()
        self.controls_area_layout.addLayout(col1_layout, 1)

        # Webcam GroupBox (Giữ nguyên)
        webcam_group = QGroupBox("Điều khiển Webcam")
        webcam_group_layout = QVBoxLayout()
        webcam_select_layout = QHBoxLayout()
        self.combo_webcam = QComboBox()
        self.btn_scan_webcam = QPushButton("Quét")
        webcam_select_layout.addWidget(QLabel("Chọn:"))
        webcam_select_layout.addWidget(self.combo_webcam, 1)
        webcam_select_layout.addWidget(self.btn_scan_webcam)
        webcam_buttons_layout = QHBoxLayout()
        self.btn_start_webcam = QPushButton("Bật Webcam")
        self.btn_stop_webcam = QPushButton("Tắt Webcam")
        self.btn_stop_webcam.setEnabled(False)
        webcam_buttons_layout.addWidget(self.btn_start_webcam)
        webcam_buttons_layout.addWidget(self.btn_stop_webcam)
        webcam_group_layout.addLayout(webcam_select_layout)
        webcam_group_layout.addLayout(webcam_buttons_layout)
        webcam_group.setLayout(webcam_group_layout)
        col1_layout.addWidget(webcam_group)

        # <<< THÊM MỚI: Audio Device Selection GroupBox >>>
        audio_group = QGroupBox("Thiết bị Âm thanh (Mic)")
        audio_layout = QHBoxLayout()
        self.combo_audio_device = QComboBox()
        self.btn_scan_audio = QPushButton("Quét Mic")
        audio_layout.addWidget(QLabel("Chọn Mic:"))
        audio_layout.addWidget(self.combo_audio_device, 1)
        audio_layout.addWidget(self.btn_scan_audio)
        audio_group.setLayout(audio_layout)
        col1_layout.addWidget(audio_group)
        # <<< /THÊM MỚI >>>

        # Recording GroupBox (Cập nhật)
        record_group = QGroupBox("Điều khiển Ghi hình (Video + Audio)") # Cập nhật tiêu đề
        record_group_layout = QVBoxLayout()
        # Save Directory Layout (Giữ nguyên)
        save_dir_layout = QHBoxLayout()
        self.lbl_save_dir = QLabel("...")
        self.lbl_save_dir.setStyleSheet("font-style: italic; border: 1px solid #ccc; padding: 2px; background-color: white;")
        self.lbl_save_dir.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.btn_select_dir = QPushButton("Chọn Thư mục Lưu")
        save_dir_layout.addWidget(QLabel("Lưu vào:"))
        save_dir_layout.addWidget(self.lbl_save_dir, 1)
        save_dir_layout.addWidget(self.btn_select_dir)
        # Record Buttons Layout (Thêm nút Reset)
        record_buttons_layout = QHBoxLayout()
        self.btn_start_record = QPushButton("Bắt đầu Ghi")
        self.btn_pause_record = QPushButton("Tạm dừng Video") # Làm rõ chỉ pause video
        self.btn_stop_save_record = QPushButton("Dừng & Lưu")
        self.btn_reset_counter = QPushButton("Reset Đếm") # <<< THÊM NÚT RESET >>>
        self.btn_start_record.setEnabled(False); self.btn_pause_record.setEnabled(False); self.btn_stop_save_record.setEnabled(False)
        record_buttons_layout.addWidget(self.btn_start_record)
        record_buttons_layout.addWidget(self.btn_pause_record)
        record_buttons_layout.addWidget(self.btn_stop_save_record)
        record_buttons_layout.addWidget(self.btn_reset_counter) # <<< THÊM VÀO LAYOUT >>>
        # Recording Status Label (Giữ nguyên)
        self.lbl_record_status = QLabel("Trạng thái: Sẵn sàng")
        self.lbl_record_status.setAlignment(Qt.AlignCenter)
        self.lbl_record_status.setFont(QFont("Arial", 11, QFont.Bold))
        # Add sub-layouts to group
        record_group_layout.addLayout(save_dir_layout)
        record_group_layout.addLayout(record_buttons_layout)
        record_group_layout.addWidget(self.lbl_record_status)
        record_group.setLayout(record_group_layout)
        col1_layout.addWidget(record_group)
        col1_layout.addStretch()

        # --- 2b. Column 2: Serial Controls & Log (Giữ nguyên) ---
        col2_layout = QVBoxLayout()
        self.controls_area_layout.addLayout(col2_layout, 1)
        serial_group = QGroupBox("Điều khiển Cổng Serial (COM)")
        # ... (Giữ nguyên cấu trúc bên trong Serial GroupBox) ...
        serial_layout_main = QVBoxLayout()
        serial_config_layout = QHBoxLayout()
        self.combo_com_port = QComboBox()
        self.combo_baud_rate = QComboBox()
        self.btn_scan_serial = QPushButton("Quét Cổng")
        self.combo_baud_rate.addItems(self.common_baud_rates)
        try:
            default_baud_index = self.common_baud_rates.index(self.default_baud_rate)
            self.combo_baud_rate.setCurrentIndex(default_baud_index)
        except ValueError:
             if len(self.common_baud_rates) > 0: self.combo_baud_rate.setCurrentIndex(0)
        serial_config_layout.addWidget(QLabel("Cổng:"))
        serial_config_layout.addWidget(self.combo_com_port, 2)
        serial_config_layout.addWidget(QLabel("Baud:"))
        serial_config_layout.addWidget(self.combo_baud_rate, 1)
        serial_config_layout.addWidget(self.btn_scan_serial)
        serial_connect_layout = QHBoxLayout()
        self.btn_connect_serial = QPushButton("Kết nối")
        self.btn_disconnect_serial = QPushButton("Ngắt Kết nối")
        self.btn_disconnect_serial.setEnabled(False)
        serial_connect_layout.addStretch()
        serial_connect_layout.addWidget(self.btn_connect_serial)
        serial_connect_layout.addWidget(self.btn_disconnect_serial)
        serial_connect_layout.addStretch()
        serial_log_layout = QVBoxLayout()
        serial_log_layout.addWidget(QLabel("Log Serial:"))
        self.serial_log = QTextEdit()
        self.serial_log.setReadOnly(True)
        self.serial_log.setFixedHeight(150) # Tăng chiều cao một chút
        self.serial_log.setFont(QFont("Consolas", 9))
        serial_log_layout.addWidget(self.serial_log)
        serial_layout_main.addLayout(serial_config_layout)
        serial_layout_main.addLayout(serial_connect_layout)
        serial_layout_main.addLayout(serial_log_layout)
        serial_group.setLayout(serial_layout_main)
        col2_layout.addWidget(serial_group)
        col2_layout.addStretch()

        # --- 3. Bottom Area: Exit Button & Status Bar (Giữ nguyên) ---
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        self.btn_exit = QPushButton("Thoát")
        bottom_layout.addWidget(self.btn_exit)
        self.main_layout.addLayout(bottom_layout)
        self.statusBar = self.statusBar()
        self.status_label = QLabel(" Sẵn sàng")
        self.statusBar.addWidget(self.status_label, 1)

        # --- Connect Signals ---
        self._connect_signals()
        print("UI Initialized and Signals Connected.")


    def _connect_signals(self):
        """Connect all UI element signals to their slots."""
        # Webcam Controls
        self.btn_scan_webcam.clicked.connect(self._scan_webcams)
        self.btn_start_webcam.clicked.connect(self._start_webcam)
        self.btn_stop_webcam.clicked.connect(self._stop_webcam)

        # <<< THÊM MỚI: Audio Controls >>>
        self.btn_scan_audio.clicked.connect(self._scan_audio_devices)
        self.combo_audio_device.currentIndexChanged.connect(self._on_audio_device_selected)

        # Recording Controls
        self.btn_select_dir.clicked.connect(self._select_save_directory)
        self.btn_start_record.clicked.connect(self._manual_start_recording)
        self.btn_pause_record.clicked.connect(self._manual_pause_recording)
        self.btn_stop_save_record.clicked.connect(self._manual_stop_save_recording)
        self.btn_reset_counter.clicked.connect(self._reset_recording_counter) # <<< KẾT NỐI RESET >>>

        # Serial Controls
        self.btn_scan_serial.clicked.connect(self._scan_serial_ports)
        self.btn_connect_serial.clicked.connect(self._connect_serial)
        self.btn_disconnect_serial.clicked.connect(self._disconnect_serial)

        # Exit Button
        self.btn_exit.clicked.connect(self.close)


    # ================== UI Update Slots & Helpers ==================

    def _log_serial(self, message):
        """Append a timestamped message to the serial log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.serial_log.append(f"[{timestamp}] {message}")
        self.serial_log.ensureCursorVisible()

    def _update_status(self, message):
        """Update the status bar message and print to console."""
        self.status_label.setText(f" {message}")
        # print(f"Status: {message}") # Giảm bớt log console trùng lặp

    def _update_status_visuals(self):
        """Update the recording status label (flashing effect)."""
        status_text = "Sẵn sàng"
        style_sheet = "color: black;" # Default style

        if self.is_recording:
            display_filename_vid = os.path.basename(self.last_video_filename) if self.last_video_filename else "..."
            display_filename_aud = os.path.basename(self.last_audio_filename) if self.last_audio_filename else "..."
            base_text = f"VID: {display_filename_vid} | AUD: {display_filename_aud}"

            self.recording_flash_state = not self.recording_flash_state
            if self.is_paused: # Chỉ trạng thái pause của video
                status_text = f"TẠM DỪNG VIDEO - {base_text}"
                style_sheet = "color: orange; font-weight: bold;"
            else:
                status_text = f"ĐANG GHI - {base_text}"
                style_sheet = "color: red; font-weight: bold;" if self.recording_flash_state else "color: darkred; font-weight: bold;"
        elif self.webcam_thread and self.webcam_thread.isRunning():
             status_text = "Webcam Bật"
             style_sheet = "color: green; font-weight: bold;"

        self.lbl_record_status.setText(f"Trạng thái: {status_text}")
        self.lbl_record_status.setStyleSheet(style_sheet)

    def _update_save_dir_label(self):
        """Update the save directory label, shortening if needed."""
        display_path = self.save_directory
        # ... (Giữ nguyên logic rút gọn đường dẫn) ...
        max_len = 50
        if len(display_path) > max_len:
            start = display_path[:max_len//2 - 2]
            end = display_path[-(max_len//2 - 1):]
            display_path = f"{start}...{end}"
        self.lbl_save_dir.setText(display_path)
        self.lbl_save_dir.setToolTip(self.save_directory)


    # ================== Device Scan Methods ==================

    def _scan_webcams(self):
        """Scan for available webcams and update the combobox."""
        self.combo_webcam.clear()
        available_webcams = []
        index = 0
        max_scan_index = 5
        print("Scanning for webcams...")
        while index < max_scan_index:
            # print(f"  Checking webcam index: {index}") # Giảm log
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            opened = cap.isOpened()
            if not opened: cap.release(); cap = cv2.VideoCapture(index); opened = cap.isOpened()

            if opened:
                cam_name = f"Webcam {index}"
                # Cố gắng lấy tên thân thiện hơn (có thể không hoạt động trên mọi OS/backend)
                try: backend_name = cap.getBackendName()
                except: backend_name = "?"
                # print(f"  Webcam {index} ({backend_name}) opened.")
                available_webcams.append((index, cam_name))
                cap.release()
                index += 1
            else:
                cap.release()
                # print(f"  Webcam index {index} failed.")
                break # Dừng quét nếu không mở được index liên tiếp

        if not available_webcams:
            self.combo_webcam.addItem("Không tìm thấy webcam")
            self.btn_start_webcam.setEnabled(False)
            self._update_status("Không tìm thấy webcam nào.")
        else:
            for idx, name in available_webcams: self.combo_webcam.addItem(name, userData=idx)
            self.btn_start_webcam.setEnabled(True)
            self._update_status(f"Tìm thấy {len(available_webcams)} webcam.")
            if len(available_webcams) > 0: self.combo_webcam.setCurrentIndex(0)

    def _scan_serial_ports(self):
        """Scan for available serial ports and update the combobox."""
        self.combo_com_port.clear()
        ports = serial.tools.list_ports.comports()
        found_ports = []
        print("Scanning for serial ports...")
        if ports:
            for port in sorted(ports, key=lambda p: p.device):
                if "COM" in port.device.upper() or "ACM" in port.device.upper() or "USB" in port.device.upper():
                     desc = f" - {port.description}" if port.description and port.description != "n/a" else ""
                     found_ports.append((port.device, f"{port.device}{desc}"))
                     # print(f"  Found Serial: {port.device}{desc}") # Giảm log

        if not found_ports:
            self.combo_com_port.addItem("Không tìm thấy cổng COM")
            self.btn_connect_serial.setEnabled(False)
            self._update_status("Không tìm thấy cổng COM nào.")
        else:
            for device, name in found_ports: self.combo_com_port.addItem(name, userData=device)
            self.btn_connect_serial.setEnabled(True)
            self._update_status(f"Tìm thấy {len(found_ports)} cổng COM.")
            if len(found_ports) > 0: self.combo_com_port.setCurrentIndex(0)

    # <<< THÊM MỚI: Hàm quét thiết bị audio >>>
    def _scan_audio_devices(self):
        """Scan for available audio input devices and update the combobox."""
        self.combo_audio_device.clear()
        available_devices = []
        print("Scanning for audio input devices...")
        try:
            devices = sd.query_devices()
            # print(f"Sounddevice found devices: {devices}") # Log chi tiết nếu cần debug
            default_input_idx = -1
            try: # Cố gắng lấy index thiết bị mặc định
                 default_input_idx = sd.default.device[0] # Index 0 là input
                 # print(f"Default input device index: {default_input_idx}")
            except Exception as e_def:
                 print(f"Could not get default input device: {e_def}")


            for i, device in enumerate(devices):
                # Chỉ lấy thiết bị có kênh đầu vào > 0
                if device.get('max_input_channels', 0) > 0:
                    host_api_info = sd.query_hostapis(device['hostapi'])
                    device_name = f"{i}: {device['name']} ({host_api_info['name']})"
                    is_default = "(Mặc định)" if i == default_input_idx else ""
                    display_name = f"{device_name} {is_default}"
                    available_devices.append({'index': i, 'name': display_name, 'samplerate': device['default_samplerate']})
                    # print(f"  Found Audio Input: {display_name}") # Giảm log

        except Exception as e:
            print(f"Error scanning audio devices: {e}", file=sys.stderr)
            self._update_status(f"Lỗi quét thiết bị âm thanh: {e}")
            QMessageBox.warning(self, "Lỗi Âm thanh", f"Không thể quét thiết bị âm thanh:\n{e}")

        if not available_devices:
            self.combo_audio_device.addItem("Không tìm thấy Mic")
            self.combo_audio_device.setEnabled(False)
            self.audio_device_index = None # Đảm bảo không có index nào được chọn
        else:
            self.combo_audio_device.setEnabled(True)
            # Thêm "Thiết bị mặc định" làm lựa chọn đầu tiên (userData=None)
            self.combo_audio_device.addItem("Thiết bị mặc định", userData=None)
            default_selected = False
            for dev_info in available_devices:
                self.combo_audio_device.addItem(dev_info['name'], userData=dev_info['index'])
                # Nếu tìm thấy thiết bị mặc định thực sự, chọn nó
                if dev_info['index'] == default_input_idx:
                    self.combo_audio_device.setCurrentText(dev_info['name']) # Chọn theo tên hiển thị
                    self.audio_device_index = dev_info['index']
                    default_selected = True

            # Nếu không có mặc định rõ ràng, chọn mục "Thiết bị mặc định" (index 0 của combo)
            if not default_selected:
                 self.combo_audio_device.setCurrentIndex(0)
                 self.audio_device_index = None # None nghĩa là dùng default của sounddevice

            self._update_status(f"Tìm thấy {len(available_devices)} thiết bị ghi âm.")
        print(f"Selected audio device index: {self.audio_device_index}")


    # <<< THÊM MỚI: Slot khi chọn thiết bị audio >>>
    def _on_audio_device_selected(self, index):
        """Update the selected audio device index when the combobox changes."""
        if index >= 0: # Đảm bảo index hợp lệ
            selected_data = self.combo_audio_device.itemData(index)
            self.audio_device_index = selected_data # Sẽ là None nếu chọn "Thiết bị mặc định"
            print(f"Audio device selection changed to index: {self.audio_device_index}")
            # Có thể cập nhật samplerate mặc định ở đây nếu muốn
            # Hoặc hiển thị thông tin thiết bị trong status bar


    # ================== Webcam Control Methods (Gần như giữ nguyên) ==================

    def _start_webcam(self):
        """Start the selected webcam."""
        if self.webcam_thread and self.webcam_thread.isRunning():
            QMessageBox.warning(self, "Thông báo", "Webcam đã đang chạy.")
            return
        selected_index = self.combo_webcam.currentIndex()
        if selected_index < 0 or self.combo_webcam.itemData(selected_index) is None:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn webcam hợp lệ.")
            return

        webcam_idx = self.combo_webcam.itemData(selected_index)
        print(f"Starting webcam {webcam_idx}...")
        self.video_frame_label.setText(f"Đang kết nối Webcam {webcam_idx}...")
        self.video_frame_label.repaint()
        self.btn_start_webcam.setEnabled(False)
        self.btn_stop_webcam.setEnabled(True)
        self.combo_webcam.setEnabled(False)
        self.btn_scan_webcam.setEnabled(False)
        self._update_status(f"Đang khởi động Webcam {webcam_idx}...")

        self.webcam_thread = WebcamThread(webcam_idx)
        self.webcam_thread.frame_ready.connect(self._update_frame)
        self.webcam_thread.error.connect(self._handle_webcam_error)
        self.webcam_thread.properties_ready.connect(self._on_webcam_properties_ready)
        self.webcam_thread.finished.connect(self._on_webcam_thread_finished)
        self.webcam_thread.start()

    def _on_webcam_properties_ready(self, width, height, fps):
        """Slot called when webcam properties are successfully retrieved."""
        if self.webcam_thread and self.sender() == self.webcam_thread:
            self.webcam_properties = {'width': width, 'height': height, 'fps': fps}
            print(f"Received webcam properties: {self.webcam_properties}")
            status_msg = f"Webcam {self.combo_webcam.currentData()} bật [{width}x{height} @ {fps:.2f} FPS]."
            self._update_status(status_msg)
            # Chỉ bật nút ghi hình khi webcam sẵn sàng
            if self.webcam_thread.isRunning():
                self.btn_start_record.setEnabled(True) # Bật nút Bắt đầu Ghi
                self.btn_pause_record.setEnabled(False)
                self.btn_stop_save_record.setEnabled(False)
                self.status_timer.start(500)

    def _stop_webcam(self):
        """Stop the running webcam thread and handle recording state."""
        if not (self.webcam_thread and self.webcam_thread.isRunning()):
             # print("Stop webcam request ignored: No webcam running.") # Giảm log
             return

        print("Stop webcam requested...")
        self.status_timer.stop()

        should_proceed_with_stop = True
        if self.is_recording:
            print("Recording is active. Confirming stop/save...")
            should_proceed_with_stop = self._confirm_and_stop_recording(
                "Webcam đang tắt.\nBạn có muốn lưu video/audio đang quay không?"
            )

        if not should_proceed_with_stop:
             print("Webcam stop cancelled by user during confirmation.")
             if self.webcam_thread and self.webcam_thread.isRunning():
                 self.status_timer.start(500)
             return

        print("Proceeding to stop webcam thread...")
        self._update_status("Đang tắt webcam...")

        if self.webcam_thread:
            # Ngắt kết nối tín hiệu webcam trước khi dừng
            signals_to_disconnect = [
                (self.webcam_thread.frame_ready, self._update_frame),
                (self.webcam_thread.error, self._handle_webcam_error),
                (self.webcam_thread.properties_ready, self._on_webcam_properties_ready),
                (self.webcam_thread.finished, self._on_webcam_thread_finished)
            ]
            for signal, slot in signals_to_disconnect:
                try: signal.disconnect(slot)
                except: pass

            self.webcam_thread.stop() # stop() bao gồm wait()

    def _on_webcam_thread_finished(self):
        """Slot called when the WebcamThread has completely finished."""
        print("Webcam thread 'finished' signal received. Resetting UI.")
        self.webcam_thread = None

        self.video_frame_label.setText("Webcam đã tắt")
        self.video_frame_label.setPixmap(QPixmap())

        self.btn_start_webcam.setEnabled(True)
        self.btn_stop_webcam.setEnabled(False)
        self.combo_webcam.setEnabled(True)
        self.btn_scan_webcam.setEnabled(True)

        # Reset recording state (quan trọng nếu webcam bị lỗi khi đang ghi)
        if self.is_recording:
             print("Warning: Webcam finished while recording was marked active. Forcing recording stop state.")
             # Không gọi hàm stop phức tạp ở đây, chỉ reset cờ và UI
             self.is_recording = False
             self.is_paused = False
             # Đảm bảo audio cũng dừng nếu webcam dừng đột ngột
             if self.audio_thread and self.audio_thread.isRunning():
                 print("Stopping associated audio thread due to webcam finish.")
                 self.audio_thread.stop()
                 if not self.audio_thread.wait(1500): print("Audio thread wait timeout during webcam finish.")
                 self.audio_thread = None
                 # Cần xử lý file audio tạm thời ở đây không? Có lẽ nên để lại file đã ghi.
             # Đảm bảo video writer đóng lại
             if self.video_writer and self.video_writer.isOpened():
                 print("Releasing video writer due to webcam finish.")
                 try: self.video_writer.release()
                 except Exception as e: print(f"Error releasing video writer: {e}")
             self.video_writer = None
             self.last_video_filename = ""
             self.last_audio_filename = ""


        self.btn_start_record.setEnabled(False) # Tắt nút ghi khi webcam tắt
        self.btn_pause_record.setEnabled(False); self.btn_pause_record.setText("Tạm dừng Video")
        self.btn_stop_save_record.setEnabled(False)

        self.status_timer.stop()
        self._update_status("Webcam đã tắt.")
        self._update_status_visuals()


    def _handle_webcam_error(self, message):
        """Handle errors emitted by the webcam thread."""
        if self.webcam_thread and self.sender() == self.webcam_thread:
             QMessageBox.critical(self, "Lỗi Webcam", message)
             self._update_status(f"Lỗi Webcam: {message}")
             print(f"Webcam Error: {message}", file=sys.stderr)
             # Thử dừng webcam một cách an toàn khi có lỗi
             self._stop_webcam() # stop_webcam đã bao gồm xử lý recording
        # else: print(f"Ignoring error from non-active webcam thread: {message}") # Giảm log


    def _update_frame(self, frame):
        """Update the video display label and write frame if recording."""
        if frame is None: return

        try:
            # Hiển thị frame (giữ nguyên logic)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            label_size = self.video_frame_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                 pixmap = QPixmap.fromImage(qt_image)
                 scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.FastTransformation)
                 self.video_frame_label.setPixmap(scaled_pixmap)
            else:
                 self.video_frame_label.setPixmap(QPixmap.fromImage(qt_image))

        except Exception as e:
            print(f"Error converting/displaying frame: {e}", file=sys.stderr)
            # Có thể dừng webcam nếu lỗi hiển thị liên tục

        # Ghi frame video nếu đang ghi và không pause
        if self.is_recording and not self.is_paused:
            writer = self.video_writer
            if writer and writer.isOpened():
                try:
                    writer.write(frame) # Ghi frame BGR gốc
                except Exception as e:
                    error_msg = f"Lỗi ghi frame video: {e}"
                    print(error_msg, file=sys.stderr)
                    self._log_serial(error_msg) # Ghi lỗi vào log serial
                    # Dừng ghi hình khi có lỗi ghi frame? Cân nhắc
                    QMessageBox.critical(self, "Lỗi Ghi Video", f"{error_msg}\nĐang dừng ghi hình.")
                    # Gọi hàm dừng an toàn, giả sử là lưu lại những gì đã có
                    self._stop_save_recording("VideoWriteError")


    # ================== Serial Control Methods (Giữ nguyên) ==================
    # ... (Các hàm _scan_serial_ports, _connect_serial, _disconnect_serial,
    # _on_serial_thread_finished, _handle_serial_error, _handle_serial_data) ...
    # Giữ nguyên nội dung các hàm này như trong code gốc của bạn
    def _connect_serial(self):
        if self.serial_thread and self.serial_thread.isRunning():
            QMessageBox.warning(self, "Thông báo", "Đã kết nối Serial.")
            return
        selected_port_index = self.combo_com_port.currentIndex()
        if selected_port_index < 0 or self.combo_com_port.itemData(selected_port_index) is None:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn cổng COM hợp lệ.")
            return
        port_name = self.combo_com_port.itemData(selected_port_index)
        selected_baud_text = self.combo_baud_rate.currentText()
        try:
            baud_rate = int(selected_baud_text)
            if baud_rate <= 0: raise ValueError("Baud rate phải dương.")
        except ValueError:
            QMessageBox.critical(self, "Lỗi Baudrate", f"Baudrate không hợp lệ: '{selected_baud_text}'.")
            return

        print(f"Connecting to Serial: {port_name} @ {baud_rate} baud...")
        log_msg = f"Đang kết nối tới {port_name} tại {baud_rate} baud..."
        self._log_serial(log_msg)
        self.serial_log.repaint()
        self._update_status(f"Đang kết nối {port_name}@{baud_rate}...")
        self.btn_connect_serial.setEnabled(False)
        self.btn_disconnect_serial.setEnabled(True)
        self.combo_com_port.setEnabled(False)
        self.combo_baud_rate.setEnabled(False)
        self.btn_scan_serial.setEnabled(False)

        self.serial_thread = SerialThread(port_name, baudrate=baud_rate)
        self.serial_thread.data_received.connect(self._handle_serial_data)
        self.serial_thread.error.connect(self._handle_serial_error)
        self.serial_thread.finished.connect(self._on_serial_thread_finished)
        self.serial_thread.start()

    def _disconnect_serial(self):
        if self.serial_thread and self.serial_thread.isRunning():
             port = self.serial_thread.port
             baud = self.serial_thread.baudrate
             print(f"Disconnecting Serial: {port} @ {baud} baud...")
             self._log_serial(f"Đang ngắt kết nối Serial ({port}@{baud})...")
             self._update_status(f"Đang ngắt kết nối Serial ({port})...")
             signals_to_disconnect = [
                 (self.serial_thread.data_received, self._handle_serial_data),
                 (self.serial_thread.error, self._handle_serial_error),
                 (self.serial_thread.finished, self._on_serial_thread_finished)
             ]
             for signal, slot in signals_to_disconnect:
                 try: signal.disconnect(slot)
                 except: pass
             self.serial_thread.stop()
        # else: print("Disconnect serial ignored: No active connection.") # Giảm log
        elif not self.serial_thread: self._on_serial_thread_finished()

    def _on_serial_thread_finished(self):
        print("Serial thread 'finished' signal received. Resetting UI.")
        was_connected = bool(self.serial_thread)
        self.serial_thread = None
        self.btn_connect_serial.setEnabled(True)
        self.btn_disconnect_serial.setEnabled(False)
        self.combo_com_port.setEnabled(True)
        self.combo_baud_rate.setEnabled(True)
        self.btn_scan_serial.setEnabled(True)
        if was_connected:
            self._log_serial("Đã ngắt kết nối Serial.")
            self._update_status("Đã ngắt kết nối Serial.")

    def _handle_serial_error(self, message):
        if self.serial_thread and self.sender() == self.serial_thread:
            log_msg = f"LỖI SERIAL: {message}"
            self._log_serial(log_msg)
            self._update_status(f"Lỗi Serial: Xem Log")
            print(f"Serial Error: {message}", file=sys.stderr)
            QMessageBox.critical(self, "Lỗi Serial", message)
            if self.serial_thread.isRunning():
                # print("Attempting disconnect due to serial error...") # Giảm log
                self._disconnect_serial()
            else:
                self._on_serial_thread_finished()
        # else: print(f"Ignoring error from non-active serial thread: {message}") # Giảm log

    def _handle_serial_data(self, data):
        """Process commands received from the serial port."""
        self._log_serial(f"Nhận: '{data}'")
        command = data.strip().upper()

        if not (self.webcam_thread and self.webcam_thread.isRunning()):
            self._log_serial("Lệnh Serial bị bỏ qua: Webcam chưa bật.")
            return

        if command == "START":
             if not self.is_recording:self._start_recording("Serial")
             elif self.is_paused:self._pause_recording("Serial") # Resume video
             else: self._log_serial("Lệnh 'START' bị bỏ qua: Đang ghi.")
        elif command == "STOP_SAVE":
            if self.is_recording: self._stop_save_recording("Serial")
            else: self._log_serial("Lệnh 'STOP_SAVE' bị bỏ qua: Chưa ghi.")
        elif command == "STOP_DISCARD":
            if self.is_recording: self._stop_discard_recording("Serial")
            else: self._log_serial("Lệnh 'STOP_DISCARD' bị bỏ qua: Chưa ghi.")
        elif command == "PAUSE": # Chỉ pause video
            if self.is_recording and not self.is_paused: self._pause_recording("Serial")
            else: self._log_serial("Lệnh 'PAUSE' bị bỏ qua: Chưa ghi video hoặc đã dừng video.")
        elif command == "RESUME": # Chỉ resume video
            if self.is_recording and self.is_paused: self._pause_recording("Serial")
            else: self._log_serial("Lệnh 'RESUME' bị bỏ qua: Chưa ghi video hoặc video đang chạy.")
        else:
            self._log_serial(f"Lệnh không xác định từ Serial: '{command}' (Gốc: '{data}')")


    # ================== Recording Control Methods (Cập nhật) ==================

    def _select_save_directory(self):
        """Open dialog to select video/audio save directory."""
        directory = QFileDialog.getExistingDirectory(self, "Chọn Thư mục Lưu Video & Audio", self.save_directory)
        if directory:
            self.save_directory = directory
            self._update_save_dir_label()
            self._update_status(f"Thư mục lưu: {self.save_directory}")
        # else: self._update_status("Việc chọn thư mục bị hủy.") # Giảm log

    def _generate_filenames(self):
        """Generate video (.mp4) and audio (.wav) filenames."""
        # 1. Tăng biến đếm TRƯỚC KHI tạo tên file
        self.recording_session_counter += 1
        counter = self.recording_session_counter

        # 2. Lấy thời gian hiện tại
        now = datetime.now()
        time_str = now.strftime("%H%M%S") # Thêm giây để tránh trùng lặp tốt hơn
        date_str = now.strftime("%d%m%Y")

        # 3. Tạo phần gốc của tên file
        base_filename = f"Loop_{counter}_{time_str}_{date_str}"

        # 4. Tạo tên file video và audio
        video_filename = f"{base_filename}.mp4"
        audio_filename = f"{base_filename}.wav" # <<< THÊM MỚI: Tên file audio

        # 5. Lưu lại tên file gần nhất
        self.last_video_filename = video_filename
        self.last_audio_filename = audio_filename # <<< THÊM MỚI: Lưu tên file audio

        print(f"Generated filenames: Video='{video_filename}', Audio='{audio_filename}'")
        return video_filename, audio_filename # Trả về cả hai tên


    def _create_video_writer(self, filepath):
        """Initialize the OpenCV VideoWriter for MP4."""
        props = self.webcam_properties
        if not all(props.values()) or props['width'] <= 0 or props['height'] <= 0 or props['fps'] <= 0:
             error_msg = f"Lỗi: Thông số webcam không hợp lệ để tạo VideoWriter: {props}"
             print(error_msg, file=sys.stderr)
             QMessageBox.critical(self, "Lỗi Ghi Video", error_msg)
             self._update_status("Lỗi thông số webcam."); return False

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = props['width']; height = props['height']; fps = props['fps']
        # Clamp FPS lại một lần nữa cho chắc
        safe_fps = max(1.0, min(120.0, fps))
        if safe_fps != fps: print(f"Warning: Clamping FPS from {fps:.2f} to {safe_fps:.2f} for VideoWriter.")

        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            print(f"Creating VideoWriter: Path='{os.path.basename(filepath)}', FourCC=mp4v, FPS={safe_fps:.2f}, Size=({width}x{height})")
            self.video_writer = cv2.VideoWriter(filepath, fourcc, safe_fps, (width, height))

            if not self.video_writer.isOpened():
                raise IOError(f"Không thể mở/tạo file video MP4: {os.path.basename(filepath)}")

            print(f"VideoWriter MP4 created successfully for {os.path.basename(filepath)}")
            return True

        except Exception as e:
            error_msg = f"Lỗi tạo VideoWriter MP4: {e}"
            QMessageBox.critical(self, "Lỗi Ghi Video", error_msg)
            self._update_status(error_msg)
            print(error_msg, file=sys.stderr)
            if self.video_writer:
                try: self.video_writer.release()
                except: pass
            self.video_writer = None
            return False

    # <<< THÊM MỚI: Hàm xử lý lỗi từ AudioThread >>>
    def _handle_audio_error(self, message):
        """Handle errors emitted by the audio thread."""
        # Chỉ xử lý lỗi từ thread audio đang hoạt động (nếu có)
        if self.audio_thread and self.sender() == self.audio_thread:
             log_msg = f"LỖI AUDIO: {message}"
             self._log_serial(log_msg) # Ghi vào log serial luôn
             self._update_status(f"Lỗi Audio: Xem Log")
             print(log_msg, file=sys.stderr)
             QMessageBox.critical(self, "Lỗi Ghi Âm Thanh", message)
             # Lỗi audio có nên dừng cả video không? Có lẽ nên.
             if self.is_recording:
                 print("Stopping recording due to critical audio error.")
                 # Gọi hàm dừng an toàn, lưu những gì đã có
                 self._stop_save_recording("AudioError")
        # else: print(f"Ignoring error from non-active audio thread: {message}") # Giảm log


    def _start_recording(self, source="Manual"):
        """Start both video and audio recording."""
        # --- Pre-checks ---
        if not (self.webcam_thread and self.webcam_thread.isRunning()):
             QMessageBox.warning(self, "Cảnh báo", "Webcam chưa bật.")
             self._log_serial(f"[{source}] Ghi thất bại: Webcam không chạy."); return
        if self.is_recording:
             msg = f"Đã {'đang ghi' if not self.is_paused else 'tạm dừng video'}."; QMessageBox.warning(self, "Cảnh báo", msg); self._log_serial(f"[{source}] Ghi thất bại: {msg}"); return
        if not self.save_directory or not os.path.isdir(self.save_directory):
            QMessageBox.warning(self, "Cảnh báo", "Thư mục lưu không hợp lệ."); self._update_status("Cần chọn thư mục lưu."); return
        if self.combo_audio_device.currentIndex() < 0 and len(sd.query_devices()) > 0: # Kiểm tra nếu có mic nhưng chưa chọn
             # Trường hợp này ít xảy ra nếu _scan_audio_devices chạy đúng
             if self.combo_audio_device.itemText(0) != "Không tìm thấy Mic":
                 QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn thiết bị âm thanh (Mic).")
                 return
             # Nếu thực sự không có mic thì cho phép ghi không tiếng? Hiện tại không cho.
             else:
                 QMessageBox.warning(self, "Cảnh báo", "Không tìm thấy thiết bị ghi âm thanh.")
                 return

        # --- Generate Filenames ---
        video_filename, audio_filename = self._generate_filenames()
        video_filepath = os.path.join(self.save_directory, video_filename)
        audio_filepath = os.path.join(self.save_directory, audio_filename) # <<< THÊM MỚI

        # --- Start Audio Recording Thread FIRST ---
        # Lý do: Nếu audio thất bại, không cần tạo video writer
        print(f"Starting AudioThread for: {audio_filename}")
        self.audio_thread = AudioThread(
            filename=audio_filepath,
            samplerate=self.audio_samplerate,
            channels=self.audio_channels,
            device=self.audio_device_index # Lấy từ combobox hoặc None (mặc định)
        )
        self.audio_thread.error.connect(self._handle_audio_error)
        # Kết nối status update nếu muốn log chi tiết hơn
        # self.audio_thread.status_update.connect(self._log_serial)
        # Kết nối finished writing nếu cần làm gì đó khi file audio đóng xong
        # self.audio_thread.finished_writing.connect(lambda fn: print(f"Confirmed audio written: {fn}"))
        self.audio_thread.start()
        # Chờ một chút xem audio có báo lỗi ngay không? (Tùy chọn, có thể làm chậm khởi động)
        # self.audio_thread.msleep(100)
        # if not self.audio_thread.isRunning(): # Kiểm tra sơ bộ xem thread có chạy không
        #    QMessageBox.critical(self, "Lỗi Ghi Âm", "Không thể khởi động luồng ghi âm thanh.")
        #    self.audio_thread = None # Dọn dẹp
        #    self.last_video_filename = "" # Xóa tên file đã tạo
        #    self.last_audio_filename = ""
        #    return # Không tiếp tục nếu audio lỗi ngay

        # --- Create Video Writer ---
        if self._create_video_writer(video_filepath):
            # --- Success: Update State & UI ---
            self.is_recording = True
            self.is_paused = False # Video không pause khi bắt đầu
            status_msg = f"Bắt đầu ghi: {os.path.basename(video_filepath)} + {os.path.basename(audio_filepath)}"
            self._update_status(status_msg)
            self._log_serial(f"Bắt đầu ghi [{source}]: Video={video_filename}, Audio={audio_filename}")

            self.btn_start_record.setEnabled(False)
            self.btn_pause_record.setEnabled(True) # Pause chỉ cho video
            self.btn_pause_record.setText("Tạm dừng Video")
            self.btn_stop_save_record.setEnabled(True)
            self._update_status_visuals()
            if not self.status_timer.isActive(): self.status_timer.start(500)
        else:
             # --- Video Writer Failure: Stop Audio Thread ---
             print("VideoWriter creation failed. Stopping audio thread...")
             if self.audio_thread and self.audio_thread.isRunning():
                 self.audio_thread.stop()
                 if not self.audio_thread.wait(1500): print("Audio thread wait timeout during video writer failure.")
                 self.audio_thread = None
                 # Xóa file audio tạm nếu có thể (thread có thể chưa kịp tạo/ghi)
                 if os.path.exists(audio_filepath):
                     try: os.remove(audio_filepath); print(f"Removed incomplete audio file: {audio_filename}")
                     except OSError as e: print(f"Error removing incomplete audio file: {e}")

             self.last_video_filename = "" # Clear generated filenames
             self.last_audio_filename = ""


    def _pause_recording(self, source="Manual"):
        """Toggle pause/resume state of VIDEO recording ONLY."""
        # Hiện tại không hỗ trợ pause audio stream một cách đơn giản và đồng bộ
        if not self.is_recording:
             self._log_serial(f"[{source}] Pause/Resume Video bị bỏ qua: Chưa ghi."); return

        self.is_paused = not self.is_paused # Toggle state video pause
        if self.is_paused:
            self.btn_pause_record.setText("Tiếp tục Video")
            status_msg = "Đã tạm dừng ghi video (audio vẫn ghi)."; log_msg = f"Tạm dừng Video [{source}]."
        else:
            self.btn_pause_record.setText("Tạm dừng Video"); status_msg = "Đã tiếp tục ghi video."; log_msg = f"Tiếp tục Video [{source}]."

        self._update_status(status_msg); self._log_serial(log_msg); self._update_status_visuals()


    def _stop_recording_base(self, action_type, source):
        """Core logic to stop video and audio recording."""
        if not self.is_recording:
             self._log_serial(f"[{source}] Dừng ({action_type}) bị bỏ qua: Chưa ghi."); return False

        print(f"Stop recording ({action_type} by {source}) requested for Video='{self.last_video_filename}', Audio='{self.last_audio_filename}'")
        original_video_filename = self.last_video_filename
        original_audio_filename = self.last_audio_filename

        self.is_recording = False
        self.is_paused = False # Reset pause state

        video_filepath_to_process = os.path.join(self.save_directory, original_video_filename) if original_video_filename else ""
        audio_filepath_to_process = os.path.join(self.save_directory, original_audio_filename) if original_audio_filename else ""

        # --- 1. Stop Audio Thread ---
        audio_stopped_cleanly = False
        if self.audio_thread:
            audio_thread_ref = self.audio_thread # Giữ tham chiếu tạm
            self.audio_thread = None # Xóa tham chiếu chính
            print("Requesting audio thread stop...")
            audio_thread_ref.stop()
            if audio_thread_ref.wait(2000): # Chờ tối đa 2 giây
                 audio_stopped_cleanly = True
                 print("Audio thread stopped cleanly.")
            else:
                 print("Warning: Audio thread did not stop cleanly within timeout.")
                 # Không terminate audio thread vì có thể làm hỏng file wav
        else:
            print("Warning: No audio thread object found during stop.")


        # --- 2. Release Video Writer ---
        video_writer_released_cleanly = False
        video_writer_was_opened = False
        release_error = None
        writer = self.video_writer
        if writer:
            self.video_writer = None # Xóa tham chiếu chính
            video_writer_was_opened = writer.isOpened()
            if video_writer_was_opened:
                print(f"Releasing VideoWriter for {original_video_filename}...")
                try:
                    writer.release()
                    video_writer_released_cleanly = True
                    print("VideoWriter released successfully.")
                except Exception as e:
                    release_error = e
                    print(f"Error releasing VideoWriter: {e}", file=sys.stderr)
            else:
                 print(f"Warning: VideoWriter for {original_video_filename} was not open when stop was requested.")
        else:
            print("Warning: No video writer object found during stop.")

        # --- 3. Process Files based on Action ---
        final_status_msg = ""
        final_log_msg = ""

        if action_type == "Save":
            # Kiểm tra xem các file có tồn tại không
            video_exists = video_filepath_to_process and os.path.exists(video_filepath_to_process) and os.path.getsize(video_filepath_to_process) > 0
            audio_exists = audio_filepath_to_process and os.path.exists(audio_filepath_to_process) and os.path.getsize(audio_filepath_to_process) > 1024 # File wav hợp lệ thường > 1KB

            # Thông báo thành công nếu cả hai file có vẻ ổn
            if video_writer_released_cleanly and audio_stopped_cleanly and video_exists and audio_exists:
                 final_status_msg = f"Đã dừng & lưu: {original_video_filename}, {original_audio_filename}"
                 final_log_msg = f"Dừng & Lưu [{source}]: Video={original_video_filename}, Audio={original_audio_filename}. (Chưa ghép)"
                 print("Video and Audio saved successfully (separate files).")
                 # <<< CHỖ ĐỂ GỌI HÀM GHÉP FILE SAU NÀY >>>
                 # self._merge_audio_video(video_filepath_to_process, audio_filepath_to_process, ...)
            else:
                 # Xử lý lỗi lưu
                 error_parts = []
                 if not video_writer_released_cleanly: error_parts.append(f"Lỗi đóng video ({release_error or 'không mở'})")
                 elif not video_exists: error_parts.append("File video không tồn tại/trống")
                 if not audio_stopped_cleanly: error_parts.append("Lỗi dừng audio")
                 elif not audio_exists: error_parts.append("File audio không tồn tại/trống")

                 error_details = ", ".join(error_parts) if error_parts else "Lỗi không xác định"
                 final_status_msg = f"LỖI LƯU ({error_details})"
                 final_log_msg = f"Dừng & Lỗi Lưu [{source}]: {error_details}. Files: V='{original_video_filename}', A='{original_audio_filename}'"
                 QMessageBox.warning(self, "Lưu Thất Bại", f"Không thể lưu video và/hoặc audio:\n{error_details}\nVideo: {original_video_filename}\nAudio: {original_audio_filename}")

        elif action_type == "Discard":
            deleted_video = False
            deleted_audio = False
            delete_video_error = None
            delete_audio_error = None

            # Xóa video nếu writer đã được release (hoặc không mở) và file tồn tại
            if video_filepath_to_process and (video_writer_released_cleanly or not video_writer_was_opened):
                 if os.path.exists(video_filepath_to_process):
                     print(f"Attempting to delete discarded video: {original_video_filename}")
                     try:
                         os.remove(video_filepath_to_process)
                         deleted_video = True
                         print("-> Deleted video.")
                     except OSError as e: delete_video_error = e; print(f"-> Error deleting video: {e}")
                 # else: print(f"Video file {original_video_filename} not found for deletion.")

            # Xóa audio nếu thread đã dừng và file tồn tại
            if audio_filepath_to_process and audio_stopped_cleanly:
                 if os.path.exists(audio_filepath_to_process):
                     print(f"Attempting to delete discarded audio: {original_audio_filename}")
                     try:
                         os.remove(audio_filepath_to_process)
                         deleted_audio = True
                         print("-> Deleted audio.")
                     except OSError as e: delete_audio_error = e; print(f"-> Error deleting audio: {e}")
                 # else: print(f"Audio file {original_audio_filename} not found for deletion.")

            # Tạo thông báo hủy
            discard_status = []
            if deleted_video: discard_status.append("Đã xóa video")
            elif delete_video_error: discard_status.append(f"Lỗi xóa video ({delete_video_error})")
            elif video_filepath_to_process: discard_status.append("Video không bị xóa") # Hoặc không tồn tại

            if deleted_audio: discard_status.append("Đã xóa audio")
            elif delete_audio_error: discard_status.append(f"Lỗi xóa audio ({delete_audio_error})")
            elif audio_filepath_to_process: discard_status.append("Audio không bị xóa")

            discard_details = ", ".join(discard_status) if discard_status else "Trạng thái hủy không xác định"
            final_status_msg = f"Đã dừng & hủy: {discard_details}"
            final_log_msg = f"Dừng & Hủy [{source}]: {discard_details}. Files: V='{original_video_filename}', A='{original_audio_filename}'"

        # --- 4. Update UI ---
        self._update_status(final_status_msg)
        self._log_serial(final_log_msg)

        # Reset filenames sau khi xử lý xong
        self.last_video_filename = ""
        self.last_audio_filename = ""

        # Reset các nút điều khiển
        webcam_can_run = bool(self.webcam_thread and self.webcam_thread.isRunning())
        self.btn_start_record.setEnabled(webcam_can_run)
        self.btn_pause_record.setEnabled(False); self.btn_pause_record.setText("Tạm dừng Video")
        self.btn_stop_save_record.setEnabled(False)
        self._update_status_visuals() # Cập nhật trạng thái text

        return True # Hàm này trả về True nếu việc dừng được thực hiện (bất kể thành công hay lỗi)


    def _stop_save_recording(self, source="Manual"):
        """Stop recording and save the video/audio files."""
        return self._stop_recording_base("Save", source)

    def _stop_discard_recording(self, source="Manual"):
        """Stop recording and discard the video/audio files."""
        return self._stop_recording_base("Discard", source)

    def _manual_start_recording(self): self._start_recording("Manual")
    def _manual_pause_recording(self): self._pause_recording("Manual")
    def _manual_stop_save_recording(self):
        self._stop_save_recording("Manual") # Nút dừng không cần xác nhận


    # ================== Application Level Methods ==================

    def _confirm_and_stop_recording(self, message):
         """Ask user whether to Save/Discard/Cancel before stopping recording."""
         if not self.is_recording: return True # Không ghi thì cho phép tiếp tục

         reply = QMessageBox.question(self, 'Xác nhận Dừng Ghi Hình', message,
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                     QMessageBox.Save)

         if reply == QMessageBox.Save:
             # Gọi hàm dừng và lưu, trả về True nếu hàm đó thực hiện việc dừng
             return self._stop_save_recording("Confirm")
         elif reply == QMessageBox.Discard:
             # Gọi hàm dừng và hủy, trả về True nếu hàm đó thực hiện việc dừng
             return self._stop_discard_recording("Confirm")
         else: # reply == QMessageBox.Cancel
            print("User cancelled stop/close action.")
            return False # Hủy hành động (ví dụ: không đóng cửa sổ)

    # <<< CẬP NHẬT: Thêm QMessageBox vào hàm reset >>>
    def _reset_recording_counter(self):
        """Reset the recording session counter back to 0 and show confirmation."""
        old_value = self.recording_session_counter
        if old_value == 0:
             QMessageBox.information(self, "Thông báo", "Bộ đếm số lần ghi đã là 0.")
             return

        reply = QMessageBox.question(self, 'Xác nhận Reset',
                                     f"Bạn có chắc muốn reset bộ đếm số lần ghi từ {old_value} về 0 không?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.recording_session_counter = 0
            msg = f"Đã reset bộ đếm số lần ghi về 0."
            print(msg)
            self._update_status(msg)
            QMessageBox.information(self, "Đã Reset", msg)
        # else: print("Reset counter cancelled.") # Giảm log

    def closeEvent(self, event):
        """Handle window close event: confirm recording stop, stop threads."""
        print("Close event triggered.")
        should_close = True

        if self.is_recording:
            print("Recording active. Confirming stop/save before closing...")
            should_close = self._confirm_and_stop_recording(
                "Ứng dụng đang đóng.\nBạn có muốn lưu video/audio đang quay không?"
            )

        if not should_close:
            print("Application close cancelled by user.")
            event.ignore()
            return

        print("Proceeding with application close...")
        self._update_status("Đang đóng ứng dụng...")
        QApplication.processEvents()
        self.status_timer.stop()

        # --- Stop Threads Gracefully ---
        print("Stopping worker threads...")
        # Stop webcam (hàm _stop_webcam đã bao gồm ngắt tín hiệu và chờ)
        if self.webcam_thread and self.webcam_thread.isRunning():
             print("Stopping webcam thread...")
             # Không cần gọi _confirm_and_stop_recording lần nữa ở đây
             self._stop_webcam() # Đã bao gồm wait

        # Stop audio (nếu chưa dừng bởi _stop_webcam hoặc _confirm)
        if self.audio_thread and self.audio_thread.isRunning():
             print("Stopping audio thread (on close)...")
             self.audio_thread.stop()
             if not self.audio_thread.wait(1500): print("Audio thread wait timeout on close.")
             self.audio_thread = None

        # Stop serial (hàm _disconnect_serial đã bao gồm ngắt tín hiệu và chờ)
        if self.serial_thread and self.serial_thread.isRunning():
             print("Stopping serial thread...")
             self._disconnect_serial() # Đã bao gồm wait

        # --- Final Video Writer Check (Safety net) ---
        # Các hàm stop ở trên nên đã xử lý cái này
        if self.video_writer and self.video_writer.isOpened():
             print("Warning: Final check releasing video writer on exit...")
             try: self.video_writer.release(); print(" -> Released.")
             except Exception as e: print(f" -> Error releasing writer on exit: {e}")
             self.video_writer = None

        print("Exiting application cleanly.")
        event.accept()


# =============================================================================
# == Application Entry Point ==
# =============================================================================
if __name__ == '__main__':
    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

import sys
import cv2
import serial
import serial.tools.list_ports
import time
import os
import sys # Cần import sys để ghi lỗi ra stderr nếu chưa có
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QTextEdit,
                             QFileDialog, QGroupBox, QMessageBox, QSizePolicy,
                             QSpacerItem)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

# =============================================================================
# == Webcam Worker Thread ==
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
        print(f"Initializing WebcamThread for index {self.webcam_index}")

    def run(self):
        print(f"WebcamThread {self.webcam_index}: Starting run loop.")
        # --- Try preferred backend (MSMF on Windows) ---
        self.cap = cv2.VideoCapture(self.webcam_index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            print(f"Webcam {self.webcam_index}: Failed with MSMF, trying default backend...")
            self.cap = cv2.VideoCapture(self.webcam_index) # Fallback to default
            if not self.cap.isOpened():
                self.error.emit(f"Không thể mở webcam {self.webcam_index} với bất kỳ backend nào.")
                self._is_running = False
                print(f"WebcamThread {self.webcam_index}: Failed to open with any backend.")
                return # Exit run method

        # --- Get properties ---
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Validate FPS
        if not (0 < self._fps < 150): # Set reasonable bounds
            print(f"Warning: Invalid FPS ({self._fps:.2f}) detected for webcam {self.webcam_index}. Defaulting to 30.0")
            self._fps = 30.0
        self.properties_ready.emit(self._width, self._height, self._fps)
        print(f"Webcam {self.webcam_index} opened successfully ({self._width}x{self._height} @ {self._fps:.2f} FPS)")

        # --- Capture loop ---
        while self._is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                # Only emit error if the loop was supposed to be running
                if self._is_running:
                    self.error.emit(f"Mất kết nối với webcam {self.webcam_index} hoặc đọc frame thất bại.")
                    print(f"WebcamThread {self.webcam_index}: Frame read failed or lost connection.")
                self._is_running = False # Ensure loop termination
                break # Exit loop

            self.msleep(10) # Small delay to prevent high CPU load (~ max 100 fps theoretically)

        # --- Cleanup ---
        if self.cap and self.cap.isOpened():
            print(f"WebcamThread {self.webcam_index}: Releasing capture...")
            self.cap.release()
        print(f"WebcamThread {self.webcam_index}: Exiting run loop.")

    def stop(self):
        """Requests the thread to stop."""
        print(f"WebcamThread {self.webcam_index}: Stop requested.")
        self._is_running = False
        # Give the run() loop time to finish based on _is_running flag
        if not self.wait(2000): # Wait up to 2 seconds
             print(f"Warning: Webcam thread {self.webcam_index} did not finish cleanly after 2s. Terminating.")
             self.terminate() # Force terminate if stuck (last resort)
             # Attempt release again after termination
             if self.cap and self.cap.isOpened():
                  try:
                      self.cap.release()
                      print(f"Webcam {self.webcam_index} capture released after termination.")
                  except Exception as e:
                      print(f"Error releasing webcam {self.webcam_index} after termination: {e}")
        else:
            print(f"WebcamThread {self.webcam_index}: Stopped successfully.")
        # Double-check release, although run() should handle it
        if self.cap and self.cap.isOpened():
             print(f"Warning: Webcam {self.webcam_index} capture still open after wait(). Releasing fallback.")
             try: self.cap.release()
             except Exception as e: print(f"Error in fallback release for webcam {self.webcam_index}: {e}")


# =============================================================================
# == Serial Worker Thread ==
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
        print(f"Initializing SerialThread: Port={self.port}, Baudrate={self.baudrate}")

    def run(self):
        print(f"SerialThread {self.port}: Starting run loop.")
        try:
            # --- Open Connection ---
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"Serial port {self.port} opened successfully at {self.baudrate} baud.")

            # --- Read Loop ---
            while self._is_running and self.serial_connection and self.serial_connection.isOpen():
                try:
                    if self.serial_connection.in_waiting > 0:
                        try:
                            # Read line, decode, strip whitespace/newlines
                            line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                            if line: # Only emit if line is not empty
                                self.data_received.emit(line)
                        except UnicodeDecodeError as ude:
                             print(f"Serial Decode Error on {self.port}: {ude}. Skipping.")
                             # Optionally clear buffer if decode errors are persistent
                             # self.serial_connection.read(self.serial_connection.in_waiting)
                # Handle Serial specific exceptions during read/wait
                except serial.SerialException as e:
                    if self._is_running:
                        self.error.emit(f"Lỗi đọc/ghi Serial ({self.port}): {e}")
                        print(f"SerialThread {self.port}: SerialException: {e}")
                    self._is_running = False # Stop loop on error
                # Handle OS level errors (e.g., device disconnected)
                except OSError as e:
                     if self._is_running:
                        self.error.emit(f"Lỗi hệ thống cổng Serial ({self.port}): {e}")
                        print(f"SerialThread {self.port}: OSError: {e}")
                     self._is_running = False
                # Handle other unexpected errors
                except Exception as e:
                     if self._is_running:
                         self.error.emit(f"Lỗi Serial không xác định ({self.port}): {e}")
                         print(f"SerialThread {self.port}: Unexpected Exception: {e}")
                     self._is_running = False

                self.msleep(50) # Small delay to yield CPU

        # Handle exceptions during initial connection attempt
        except serial.SerialException as e:
            self.error.emit(f"Không thể mở cổng Serial {self.port} tại {self.baudrate} baud: {e}")
            print(f"SerialThread {self.port}: Failed to open port: {e}")
        except Exception as e:
             self.error.emit(f"Lỗi khởi tạo Serial không xác định ({self.port}): {e}")
             print(f"SerialThread {self.port}: Failed to initialize: {e}")
             self._is_running = False # Ensure loop doesn't start if init fails

        # --- Cleanup ---
        finally:
            if self.serial_connection and self.serial_connection.isOpen():
                try:
                    print(f"SerialThread {self.port}: Closing serial port in finally block...")
                    self.serial_connection.close()
                except Exception as e:
                     print(f"Error closing serial port {self.port} during run cleanup: {e}")
            print(f"SerialThread ({self.port}) exiting run loop.")

    def stop(self):
        """Requests the thread to stop."""
        print(f"SerialThread {self.port}: Stop requested.")
        self._is_running = False
        # Close the port proactively if open, which might help readline() exit faster
        if self.serial_connection and self.serial_connection.isOpen():
            try:
                 print(f"SerialThread {self.port}: Closing port from stop()...")
                 self.serial_connection.close()
            except Exception as e:
                print(f"Error closing serial port {self.port} in stop(): {e}")
        # Wait for the run loop to finish
        if not self.wait(1500): # Wait 1.5 seconds
            print(f"Warning: Serial thread ({self.port}) did not finish cleanly after 1.5s. Terminating.")
            self.terminate()
        else:
             print(f"SerialThread ({self.port}) stopped successfully.")


# =============================================================================
# == Main Application Window ==
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Giám sát Webcam và Điều khiển Serial (v1.0)")
        self.setGeometry(100, 100, 950, 750) # Initial size

        # --- State Variables ---
        self.webcam_thread = None
        self.serial_thread = None
        self.video_writer = None
        self.is_recording = False
        self.is_paused = False
        self.save_directory = os.getcwd() # Default to current working directory
        # self.current_frame = None # No longer needed as frame processed directly
        self.webcam_properties = {'width': None, 'height': None, 'fps': None}
        self.last_video_filename = "" # Store filename being recorded/just saved

        self.recording_session_counter = 0

        # --- Timers ---
        self.status_timer = QTimer(self) # For flashing recording status
        self.status_timer.timeout.connect(self._update_status_visuals)
        self.recording_flash_state = False

        # --- Constants ---
        self.common_baud_rates = ["9600", "19200", "38400", "57600", "115200", "250000", "4800", "2400"] # Added 250k
        self.default_baud_rate = "9600"

        # --- Initialize UI ---
        self._init_ui()

        # --- Initial Scans & UI Updates ---
        self._scan_webcams()
        self._scan_serial_ports()
        self._update_save_dir_label() # Show initial save directory

        print("MainWindow initialized.")


    def _init_ui(self):
        """Build the user interface."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # --- 1. Video Display Area ---
        self.video_frame_label = QLabel("Chưa bật Webcam")
        self.video_frame_label.setAlignment(Qt.AlignCenter)
        self.video_frame_label.setFont(QFont("Arial", 16))
        self.video_frame_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
        self.video_frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_frame_label.setMinimumSize(640, 480) # Suggest a minimum size
        self.main_layout.addWidget(self.video_frame_label, 1) # Make video area stretch more

        # --- 2. Controls Area (Horizontal Layout) ---
        self.controls_area_widget = QWidget()
        self.controls_area_layout = QHBoxLayout(self.controls_area_widget)
        # Set stretch factors for columns (e.g., make them equal width)
        # self.controls_area_layout.setStretchFactor(col1_layout, 1)
        # self.controls_area_layout.setStretchFactor(col2_layout, 1)
        self.main_layout.addWidget(self.controls_area_widget) # Add controls area below video

        # --- 2a. Column 1: Webcam & Recording Controls ---
        col1_layout = QVBoxLayout()
        self.controls_area_layout.addLayout(col1_layout, 1) # Add column, stretch factor 1

        # Webcam GroupBox
        webcam_group = QGroupBox("Điều khiển Webcam")
        webcam_group_layout = QVBoxLayout() # Vertical layout inside group
        # Layout for webcam selection and scan button
        webcam_select_layout = QHBoxLayout()
        self.combo_webcam = QComboBox()
        self.btn_scan_webcam = QPushButton("Quét")
        webcam_select_layout.addWidget(QLabel("Chọn:"))
        webcam_select_layout.addWidget(self.combo_webcam, 1) # Combobox stretches
        webcam_select_layout.addWidget(self.btn_scan_webcam)
        # Layout for Start/Stop buttons
        webcam_buttons_layout = QHBoxLayout()
        self.btn_start_webcam = QPushButton("Bật Webcam")
        self.btn_stop_webcam = QPushButton("Tắt Webcam")
        self.btn_stop_webcam.setEnabled(False) # Initially disabled
        webcam_buttons_layout.addWidget(self.btn_start_webcam)
        webcam_buttons_layout.addWidget(self.btn_stop_webcam)
        # Add sub-layouts to group layout
        webcam_group_layout.addLayout(webcam_select_layout)
        webcam_group_layout.addLayout(webcam_buttons_layout)
        webcam_group.setLayout(webcam_group_layout)
        col1_layout.addWidget(webcam_group) # Add group to column 1

        # Recording GroupBox
        record_group = QGroupBox("Điều khiển Ghi hình")
        record_group_layout = QVBoxLayout()
        # Layout for Save Directory
        save_dir_layout = QHBoxLayout()
        self.lbl_save_dir = QLabel("...") # Placeholder, updated in init
        self.lbl_save_dir.setStyleSheet("font-style: italic; border: 1px solid #ccc; padding: 2px; background-color: white;")
        self.lbl_save_dir.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred) # Allow label to expand
        self.btn_select_dir = QPushButton("Chọn Thư mục Lưu")
        save_dir_layout.addWidget(QLabel("Lưu vào:"))
        save_dir_layout.addWidget(self.lbl_save_dir, 1) # Label stretches
        save_dir_layout.addWidget(self.btn_select_dir)
        # Layout for Record Buttons
        record_buttons_layout = QHBoxLayout()
        self.btn_start_record = QPushButton("Bắt đầu Ghi")
        self.btn_pause_record = QPushButton("Tạm dừng")
        self.btn_stop_save_record = QPushButton("Dừng & Lưu")
        # Initially disabled, enabled when webcam ready
        self.btn_start_record.setEnabled(False); self.btn_pause_record.setEnabled(False); self.btn_stop_save_record.setEnabled(False)
        record_buttons_layout.addWidget(self.btn_start_record)
        record_buttons_layout.addWidget(self.btn_pause_record)
        record_buttons_layout.addWidget(self.btn_stop_save_record)
        # Recording Status Label
        self.lbl_record_status = QLabel("Trạng thái: Sẵn sàng")
        self.lbl_record_status.setAlignment(Qt.AlignCenter)
        self.lbl_record_status.setFont(QFont("Arial", 11, QFont.Bold))
        # Add sub-layouts to group layout
        record_group_layout.addLayout(save_dir_layout)
        record_group_layout.addLayout(record_buttons_layout)
        record_group_layout.addWidget(self.lbl_record_status)
        record_group.setLayout(record_group_layout)
        col1_layout.addWidget(record_group) # Add group to column 1
        col1_layout.addStretch() # Push groups to the top of column 1

        # --- 2b. Column 2: Serial Controls & Log ---
        col2_layout = QVBoxLayout()
        self.controls_area_layout.addLayout(col2_layout, 1) # Add column, stretch factor 1

        # Serial GroupBox
        serial_group = QGroupBox("Điều khiển Cổng Serial (COM)")
        serial_layout_main = QVBoxLayout() # Main vertical layout for serial group
        # Layout for Port/Baud selection and Scan button
        serial_config_layout = QHBoxLayout()
        self.combo_com_port = QComboBox()
        self.combo_baud_rate = QComboBox() # Baud rate selector
        self.btn_scan_serial = QPushButton("Quét Cổng")
        # Populate baud rate combobox
        self.combo_baud_rate.addItems(self.common_baud_rates)
        try:
            default_baud_index = self.common_baud_rates.index(self.default_baud_rate)
            self.combo_baud_rate.setCurrentIndex(default_baud_index)
        except ValueError: # Handle case where default isn't in the list
             if len(self.common_baud_rates) > 0: self.combo_baud_rate.setCurrentIndex(0)
        # Add widgets to config layout
        serial_config_layout.addWidget(QLabel("Cổng:"))
        serial_config_layout.addWidget(self.combo_com_port, 2) # Give more stretch
        serial_config_layout.addWidget(QLabel("Baud:"))
        serial_config_layout.addWidget(self.combo_baud_rate, 1)
        serial_config_layout.addWidget(self.btn_scan_serial)
        # Layout for Connect/Disconnect buttons (centered or right-aligned)
        serial_connect_layout = QHBoxLayout()
        self.btn_connect_serial = QPushButton("Kết nối")
        self.btn_disconnect_serial = QPushButton("Ngắt Kết nối")
        self.btn_disconnect_serial.setEnabled(False) # Initially disabled
        serial_connect_layout.addStretch() # Push buttons to center/right
        serial_connect_layout.addWidget(self.btn_connect_serial)
        serial_connect_layout.addWidget(self.btn_disconnect_serial)
        serial_connect_layout.addStretch()
        # Layout for Serial Log
        serial_log_layout = QVBoxLayout()
        serial_log_layout.addWidget(QLabel("Log Serial:"))
        self.serial_log = QTextEdit()
        self.serial_log.setReadOnly(True)
        self.serial_log.setFixedHeight(120) # Set fixed height for log area
        self.serial_log.setFont(QFont("Consolas", 9)) # Monospaced font for logs
        serial_log_layout.addWidget(self.serial_log)
        # Add sub-layouts to group layout
        serial_layout_main.addLayout(serial_config_layout)
        serial_layout_main.addLayout(serial_connect_layout)
        serial_layout_main.addLayout(serial_log_layout)
        serial_group.setLayout(serial_layout_main)
        col2_layout.addWidget(serial_group) # Add group to column 2
        col2_layout.addStretch() # Push group to the top of column 2

        # --- 3. Bottom Area: Exit Button & Status Bar ---
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch() # Push exit button to the right
        self.btn_exit = QPushButton("Thoát")
        bottom_layout.addWidget(self.btn_exit)
        self.main_layout.addLayout(bottom_layout) # Add bottom layout

        # Status Bar
        self.statusBar = self.statusBar()
        self.status_label = QLabel(" Sẵn sàng")
        self.statusBar.addWidget(self.status_label, 1) # Label stretches

        # --- Connect Signals ---
        self._connect_signals()
        print("UI Initialized and Signals Connected.")


    def _connect_signals(self):
        """Connect all UI element signals to their slots."""
        # Webcam Controls
        self.btn_scan_webcam.clicked.connect(self._scan_webcams)
        self.btn_start_webcam.clicked.connect(self._start_webcam)
        self.btn_stop_webcam.clicked.connect(self._stop_webcam)
        # Recording Controls
        self.btn_select_dir.clicked.connect(self._select_save_directory)
        self.btn_start_record.clicked.connect(self._manual_start_recording)
        self.btn_pause_record.clicked.connect(self._manual_pause_recording)
        self.btn_stop_save_record.clicked.connect(self._manual_stop_save_recording) # Direct save, no confirm
        # Serial Controls
        self.btn_scan_serial.clicked.connect(self._scan_serial_ports)
        self.btn_connect_serial.clicked.connect(self._connect_serial)
        self.btn_disconnect_serial.clicked.connect(self._disconnect_serial)
        # Exit Button
        self.btn_exit.clicked.connect(self.close) # Connect to main window's close method


    # ================== UI Update Slots & Helpers ==================

    def _log_serial(self, message):
        """Append a timestamped message to the serial log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.serial_log.append(f"[{timestamp}] {message}")
        self.serial_log.ensureCursorVisible() # Auto-scroll

    def _update_status(self, message):
        """Update the status bar message and print to console."""
        self.status_label.setText(f" {message}")
        print(f"Status: {message}") # Console log for debugging

    def _update_status_visuals(self):
        """Update the recording status label (flashing effect)."""
        status_text = "Sẵn sàng"
        style_sheet = "color: black;" # Default style

        if self.is_recording:
            # Determine filename part for display
            display_filename = os.path.basename(self.last_video_filename) if self.last_video_filename else "..."
            # Toggle flash state
            self.recording_flash_state = not self.recording_flash_state
            # Set text and style based on paused state and flash state
            base_text_recording = f"ĐANG GHI: {display_filename}"
            base_text_paused = f"TẠM DỪNG: {display_filename}"
            if self.is_paused:
                status_text = base_text_paused
                style_sheet = "color: orange; font-weight: bold;" # Orange for paused
            else:
                status_text = base_text_recording
                # Red flashing for recording
                style_sheet = "color: red; font-weight: bold;" if self.recording_flash_state else "color: darkred; font-weight: bold;"
        elif self.webcam_thread and self.webcam_thread.isRunning():
             # Green if webcam is on but not recording
             status_text = "Webcam Bật"
             style_sheet = "color: green; font-weight: bold;"
        # else: Keep default "Sẵn sàng"

        self.lbl_record_status.setText(f"Trạng thái: {status_text}")
        self.lbl_record_status.setStyleSheet(style_sheet)

    def _update_save_dir_label(self):
        """Update the save directory label, shortening if needed."""
        display_path = self.save_directory
        max_len = 50 # Max characters to display in label
        if len(display_path) > max_len:
            # Show beginning and end parts
            start = display_path[:max_len//2 - 2]
            end = display_path[-(max_len//2 - 1):]
            display_path = f"{start}...{end}"
        self.lbl_save_dir.setText(display_path)
        self.lbl_save_dir.setToolTip(self.save_directory) # Show full path on hover


    # ================== Webcam Control Methods ==================

    def _scan_webcams(self):
        """Scan for available webcams and update the combobox."""
        self.combo_webcam.clear()
        available_webcams = []
        index = 0
        max_scan_index = 5 # Limit scanning to first few indices
        print("Scanning for webcams...")
        while index < max_scan_index:
             print(f"  Checking index: {index}")
             cap = cv2.VideoCapture(index, cv2.CAP_MSMF) # Prefer MSMF
             opened = cap.isOpened()
             if not opened: cap.release(); cap = cv2.VideoCapture(index); opened = cap.isOpened() # Try default

             if opened:
                 cam_name = f"Webcam {index}" # Basic naming
                 available_webcams.append((index, cam_name))
                 cap.release() # IMPORTANT: Release after check
                 index += 1
             else:
                 cap.release() # IMPORTANT: Release even if not opened
                 # Stop scanning if index 0 fails or if subsequent indices fail
                 if index == 0: print("  Webcam index 0 failed, likely no cameras."); break
                 else: print(f"  Webcam index {index} failed, stopping scan."); break

        if not available_webcams:
            self.combo_webcam.addItem("Không tìm thấy webcam")
            self.btn_start_webcam.setEnabled(False)
            self._update_status("Không tìm thấy webcam nào.")
        else:
            for idx, name in available_webcams: self.combo_webcam.addItem(name, userData=idx)
            self.btn_start_webcam.setEnabled(True)
            self._update_status(f"Tìm thấy {len(available_webcams)} webcam.")
            if len(available_webcams) > 0: self.combo_webcam.setCurrentIndex(0) # Select first one


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

        # Update UI to show connection attempt
        self.video_frame_label.setText(f"Đang kết nối Webcam {webcam_idx}...")
        self.video_frame_label.repaint() # Force UI update
        self.btn_start_webcam.setEnabled(False)
        self.btn_stop_webcam.setEnabled(True)
        self.combo_webcam.setEnabled(False)
        self.btn_scan_webcam.setEnabled(False)
        self._update_status(f"Đang khởi động Webcam {webcam_idx}...")

        # --- Create and start thread ---
        self.webcam_thread = WebcamThread(webcam_idx)
        # Connect signals from thread to main window slots
        self.webcam_thread.frame_ready.connect(self._update_frame)
        self.webcam_thread.error.connect(self._handle_webcam_error)
        self.webcam_thread.properties_ready.connect(self._on_webcam_properties_ready)
        self.webcam_thread.finished.connect(self._on_webcam_thread_finished) # Important for cleanup
        self.webcam_thread.start()


    def _on_webcam_properties_ready(self, width, height, fps):
        """Slot called when webcam properties are successfully retrieved."""
        # Ensure this slot is responding to the current thread
        if self.webcam_thread and self.sender() == self.webcam_thread:
            self.webcam_properties = {'width': width, 'height': height, 'fps': fps}
            print(f"Received webcam properties: {self.webcam_properties}")
            status_msg = f"Webcam {self.combo_webcam.currentData()} bật [{width}x{height} @ {fps:.2f} FPS]."
            self._update_status(status_msg)
            # Enable recording buttons ONLY if thread is still running
            if self.webcam_thread.isRunning():
                self.btn_start_record.setEnabled(True)
                self.btn_pause_record.setEnabled(False) # Only enabled after start
                self.btn_stop_save_record.setEnabled(False) # Only enabled after start
                self.status_timer.start(500) # Start status flashing timer


    def _stop_webcam(self):
        """Stop the running webcam thread and handle recording state."""
        if not (self.webcam_thread and self.webcam_thread.isRunning()):
             print("Stop webcam request ignored: No webcam running.")
             return # Exit if no thread to stop

        print("Stop webcam requested...")
        self.status_timer.stop() # Stop visual updates first

        # --- Handle ongoing recording ---
        should_proceed_with_stop = True
        if self.is_recording:
            print("Recording is active. Confirming stop/save...")
            # Ask user what to do with the recording
            should_proceed_with_stop = self._confirm_and_stop_recording(
                "Webcam đang tắt.\nBạn có muốn lưu video đang quay không?"
            )

        if not should_proceed_with_stop:
             print("Webcam stop cancelled by user during confirmation.")
             # Restart timer if webcam continues
             if self.webcam_thread and self.webcam_thread.isRunning():
                 self.status_timer.start(500)
             return # Abort webcam stop process

        # --- Proceed with stopping the thread ---
        print("Proceeding to stop webcam thread...")
        self._update_status("Đang tắt webcam...")

        if self.webcam_thread:
            # Disconnect signals *before* calling stop to prevent potential issues
            # if the thread emits signals during shutdown.
            signals_to_disconnect = [
                (self.webcam_thread.frame_ready, self._update_frame),
                (self.webcam_thread.error, self._handle_webcam_error),
                (self.webcam_thread.properties_ready, self._on_webcam_properties_ready),
                (self.webcam_thread.finished, self._on_webcam_thread_finished)
            ]
            for signal, slot in signals_to_disconnect:
                try: signal.disconnect(slot)
                except TypeError: pass # Ignore if already disconnected
                except Exception as e: print(f"Error disconnecting {signal}: {e}")

            # Request thread stop (this includes wait())
            self.webcam_thread.stop()
            # Note: _on_webcam_thread_finished will handle final UI reset


    def _on_webcam_thread_finished(self):
        """Slot called when the WebcamThread has completely finished."""
        print("Webcam thread 'finished' signal received. Resetting UI.")
        self.webcam_thread = None # Clear the thread reference

        # Reset video display
        self.video_frame_label.setText("Webcam đã tắt")
        self.video_frame_label.setPixmap(QPixmap()) # Clear image

        # Reset webcam buttons/combo
        self.btn_start_webcam.setEnabled(True)
        self.btn_stop_webcam.setEnabled(False)
        self.combo_webcam.setEnabled(True)
        self.btn_scan_webcam.setEnabled(True)

        # Reset recording state and buttons
        self.is_recording = False
        self.is_paused = False
        self.btn_start_record.setEnabled(False)
        self.btn_pause_record.setEnabled(False); self.btn_pause_record.setText("Tạm dừng")
        self.btn_stop_save_record.setEnabled(False)

        # Final cleanup of video writer if somehow left open
        if self.video_writer and self.video_writer.isOpened():
            print("Warning: Video writer detected open during webcam finish cleanup. Releasing.")
            try: self.video_writer.release()
            except Exception as e: print(f"Error releasing orphaned video writer: {e}")
        self.video_writer = None
        self.last_video_filename = ""

        # Ensure status timer is stopped and update UI
        self.status_timer.stop()
        self._update_status("Webcam đã tắt.")
        self._update_status_visuals() # Update recording status label


    def _handle_webcam_error(self, message):
        """Handle errors emitted by the webcam thread."""
        # Only process error if it's from the currently active thread
        if self.webcam_thread and self.sender() == self.webcam_thread:
             QMessageBox.critical(self, "Lỗi Webcam", message)
             self._update_status(f"Lỗi Webcam: {message}")
             print(f"Webcam Error: {message}")
             # Try to gracefully stop the webcam thread on error
             self._stop_webcam()
        else:
            print(f"Ignoring error from non-active webcam thread: {message}")


    def _update_frame(self, frame):
        """Update the video display label with a new frame."""
        if frame is None: return # Ignore None frames

        try:
            # 1. Convert Color Space (BGR to RGB) - Fast
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            # 2. Create QImage - Relatively Fast
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 3. Scale QPixmap - Can be slower depending on size/method
            label_size = self.video_frame_label.size()
            # Only scale if label has valid dimensions
            if label_size.width() > 0 and label_size.height() > 0:
                 pixmap = QPixmap.fromImage(qt_image)
                 # Qt.FastTransformation is generally faster than SmoothTransformation
                 scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.FastTransformation)
                 self.video_frame_label.setPixmap(scaled_pixmap)
            else:
                 # If label size is invalid, show original size (might be large)
                 self.video_frame_label.setPixmap(QPixmap.fromImage(qt_image))

        except Exception as e:
            # Avoid flooding console if errors are continuous
            # Use time.time() based throttling if needed
            print(f"Error converting/displaying frame: {e}")
            # Optionally, indicate error on the label itself
            # self.video_frame_label.setText("Lỗi hiển thị Frame")

        # --- Write frame to video if recording ---
        # This happens in the GUI thread - potentially blocks UI if write is slow!
        # For high FPS or slow storage, moving write to a separate thread is better.
        if self.is_recording and not self.is_paused:
            writer = self.video_writer # Local reference for safety
            if writer and writer.isOpened():
                try:
                    writer.write(frame) # Write the original BGR frame
                except Exception as e:
                    # Handle write error
                    error_msg = f"Lỗi ghi frame video: {e}"
                    self._log_serial(error_msg) # Log to serial console
                    print(error_msg, file=sys.stderr) # Print to standard error
                    # Stop recording to prevent continuous errors?
                    # QMessageBox.warning(self, "Lỗi Ghi Video", f"{error_msg}\nĐang dừng ghi hình.")
                    # self._stop_save_recording("WriteError")


    # ================== Serial Control Methods ==================

    def _scan_serial_ports(self):
        """Scan for available serial ports and update the combobox."""
        self.combo_com_port.clear()
        ports = serial.tools.list_ports.comports()
        found_ports = []
        print("Scanning for serial ports...")
        if ports:
            for port in sorted(ports, key=lambda p: p.device):
                 # Simple filter: check for common identifiers (adjust as needed)
                 if "COM" in port.device.upper() or "ACM" in port.device.upper() or "USB" in port.device.upper():
                     desc = f" - {port.description}" if port.description and port.description != "n/a" else ""
                     found_ports.append((port.device, f"{port.device}{desc}"))
                     print(f"  Found: {port.device}{desc}")

        if not found_ports:
            self.combo_com_port.addItem("Không tìm thấy cổng COM")
            self.btn_connect_serial.setEnabled(False)
            self._update_status("Không tìm thấy cổng COM nào.")
        else:
            for device, name in found_ports: self.combo_com_port.addItem(name, userData=device)
            self.btn_connect_serial.setEnabled(True)
            self._update_status(f"Tìm thấy {len(found_ports)} cổng COM.")
            if len(found_ports) > 0: self.combo_com_port.setCurrentIndex(0) # Select first one


    def _connect_serial(self):
        """Connect to the selected serial port with the selected baud rate."""
        if self.serial_thread and self.serial_thread.isRunning():
            QMessageBox.warning(self, "Thông báo", "Đã kết nối Serial.")
            return

        # --- Get Port ---
        selected_port_index = self.combo_com_port.currentIndex()
        if selected_port_index < 0 or self.combo_com_port.itemData(selected_port_index) is None:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn cổng COM hợp lệ.")
            return
        port_name = self.combo_com_port.itemData(selected_port_index)

        # --- Get Baud Rate ---
        selected_baud_text = self.combo_baud_rate.currentText()
        try:
            baud_rate = int(selected_baud_text)
            if baud_rate <= 0: raise ValueError("Baud rate phải dương.")
        except ValueError:
            QMessageBox.critical(self, "Lỗi Baudrate", f"Baudrate không hợp lệ: '{selected_baud_text}'.")
            return

        print(f"Connecting to Serial: {port_name} @ {baud_rate} baud...")
        # Update UI during connection attempt
        log_msg = f"Đang kết nối tới {port_name} tại {baud_rate} baud..."
        self._log_serial(log_msg)
        self.serial_log.repaint() # Update log view immediately
        self._update_status(f"Đang kết nối {port_name}@{baud_rate}...")
        self.btn_connect_serial.setEnabled(False)
        self.btn_disconnect_serial.setEnabled(True) # Enable disconnect
        self.combo_com_port.setEnabled(False)
        self.combo_baud_rate.setEnabled(False) # Disable baud selection
        self.btn_scan_serial.setEnabled(False)

        # --- Create and start thread ---
        self.serial_thread = SerialThread(port_name, baudrate=baud_rate)
        self.serial_thread.data_received.connect(self._handle_serial_data)
        self.serial_thread.error.connect(self._handle_serial_error)
        self.serial_thread.finished.connect(self._on_serial_thread_finished) # Connect finished
        self.serial_thread.start()


    def _disconnect_serial(self):
        """Disconnect the currently active serial connection."""
        if self.serial_thread and self.serial_thread.isRunning():
             port = self.serial_thread.port # Get info before stopping
             baud = self.serial_thread.baudrate
             print(f"Disconnecting Serial: {port} @ {baud} baud...")
             self._log_serial(f"Đang ngắt kết nối Serial ({port}@{baud})...")
             self._update_status(f"Đang ngắt kết nối Serial ({port})...")

             # Disconnect signals first
             signals_to_disconnect = [
                 (self.serial_thread.data_received, self._handle_serial_data),
                 (self.serial_thread.error, self._handle_serial_error),
                 (self.serial_thread.finished, self._on_serial_thread_finished)
             ]
             for signal, slot in signals_to_disconnect:
                 try: signal.disconnect(slot)
                 except: pass # Ignore errors

             # Request thread stop (includes wait())
             self.serial_thread.stop()
             # _on_serial_thread_finished will reset the UI
        else:
             print("Disconnect serial ignored: No active connection.")
             # Ensure UI is in disconnected state if no thread object exists
             if not self.serial_thread: self._on_serial_thread_finished()


    def _on_serial_thread_finished(self):
        """Slot called when the SerialThread has completely finished."""
        print("Serial thread 'finished' signal received. Resetting UI.")
        was_connected = bool(self.serial_thread) # Check if it was a real disconnect
        self.serial_thread = None # Clear reference

        # Reset UI elements to disconnected state
        self.btn_connect_serial.setEnabled(True)
        self.btn_disconnect_serial.setEnabled(False)
        self.combo_com_port.setEnabled(True)
        self.combo_baud_rate.setEnabled(True) # Re-enable baud selection
        self.btn_scan_serial.setEnabled(True)

        if was_connected:
            self._log_serial("Đã ngắt kết nối Serial.")
            self._update_status("Đã ngắt kết nối Serial.")


    def _handle_serial_error(self, message):
        """Handle errors emitted by the serial thread."""
        # Only handle if from the current, active thread
        if self.serial_thread and self.sender() == self.serial_thread:
            log_msg = f"LỖI SERIAL: {message}"
            self._log_serial(log_msg)
            self._update_status(f"Lỗi Serial: Xem Log")
            print(f"Serial Error: {message}", file=sys.stderr)
            QMessageBox.critical(self, "Lỗi Serial", message)
            # Attempt graceful disconnect if thread hasn't stopped itself
            if self.serial_thread.isRunning():
                print("Attempting disconnect due to serial error...")
                self._disconnect_serial()
            else:
                # Ensure UI reset if thread stopped itself
                self._on_serial_thread_finished()
        else:
             print(f"Ignoring error from non-active serial thread: {message}")


    def _handle_serial_data(self, data):
        """Process commands received from the serial port."""
        self._log_serial(f"Nhận: '{data}'")
        command = data.strip().upper() # Normalize command

        # Require webcam to be running to process recording commands
        if not (self.webcam_thread and self.webcam_thread.isRunning()):
            self._log_serial("Lệnh Serial bị bỏ qua: Webcam chưa bật.")
            return

        # --- Process known commands ---
        if command == "START":
             if not self.is_recording:self._start_recording("Serial")
             elif self.is_paused:self._pause_recording("Serial") # Treat as Resume
             else: self._log_serial("Lệnh 'START' bị bỏ qua: Đang ghi.")
        elif command == "STOP_SAVE": # Auto-save without confirmation
            if self.is_recording: self._stop_save_recording("Serial")
            else: self._log_serial("Lệnh 'STOP_SAVE' bị bỏ qua: Chưa ghi.")
        elif command == "STOP_DISCARD":
            if self.is_recording: self._stop_discard_recording("Serial")
            else: self._log_serial("Lệnh 'STOP_DISCARD' bị bỏ qua: Chưa ghi.")
        elif command == "PAUSE":
            if self.is_recording and not self.is_paused: self._pause_recording("Serial")
            else: self._log_serial("Lệnh 'PAUSE' bị bỏ qua: Chưa ghi hoặc đã dừng.")
        elif command == "RESUME":
            if self.is_recording and self.is_paused: self._pause_recording("Serial") # Treat as Un-pause
            else: self._log_serial("Lệnh 'RESUME' bị bỏ qua: Chưa ghi hoặc đang chạy.")
        else:
            self._log_serial(f"Lệnh không xác định từ Serial: '{command}' (Gốc: '{data}')")


    # ================== Recording Control Methods ==================

    def _select_save_directory(self):
        """Open dialog to select video save directory."""
        directory = QFileDialog.getExistingDirectory(self, "Chọn Thư mục Lưu Video", self.save_directory)
        if directory:
            self.save_directory = directory
            self._update_save_dir_label()
            self._update_status(f"Thư mục lưu: {self.save_directory}")
        else:
             self._update_status("Việc chọn thư mục bị hủy.")


    def _generate_video_filename(self):
        """Generate a filename using the format Loop_N_HHMM_DDMMYYYY.mp4."""
        # 1. Tăng biến đếm số lần ghi LÊN TRƯỚC KHI tạo tên file
        self.recording_session_counter += 1

        # 2. Lấy thời gian hiện tại
        now = datetime.now()

        # 3. Định dạng chuỗi thời gian và ngày theo yêu cầu
        # Định dạng HHMM (Giờ Phút)
        time_str = now.strftime("%H%M")
        # Định dạng DDMMYYYY (Ngày Tháng Năm)
        date_str = now.strftime("%d%m%Y")

        # 4. Tạo tên file hoàn chỉnh
        # Ví dụ: Loop_1_1530_25122023.mp4
        filename = f"Loop_{self.recording_session_counter}_{time_str}_{date_str}.mp4"

        # 5. Lưu lại tên file gần nhất cho việc hiển thị và trả về
        self.last_video_filename = filename # Lưu để hiển thị trạng thái
        print(f"Generated filename: {filename}") # Log tên file đã tạo (tùy chọn)
        return filename


    def _create_video_writer(self, filepath):
        """Initialize the OpenCV VideoWriter for MP4."""
        # --- Validate Webcam Properties ---
        props = self.webcam_properties
        if not all(props.values()) or props['width'] <= 0 or props['height'] <= 0 or props['fps'] <= 0:
             error_msg = "Lỗi: Thông số webcam (kích thước/FPS) không hợp lệ."
             QMessageBox.critical(self, "Lỗi Ghi Video", f"{error_msg}\n{props}")
             self._update_status(error_msg); return False

        # --- Define Codec (MP4) ---
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Common MP4 codec
        # Alternatives if mp4v fails (may need ffmpeg support in OpenCV build):
        # fourcc = cv2.VideoWriter_fourcc(*'avc1') # H.264
        # fourcc = cv2.VideoWriter_fourcc(*'H264') # Another H.264 alias

        width = props['width']; height = props['height']; fps = props['fps']
        # Re-validate FPS just before creation
        if not (0 < fps <= 120): print(f"Warning: Invalid FPS ({fps}) at writer creation. Clamping to 30.0"); fps = 30.0

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # Log creation parameters
            print(f"Creating VideoWriter: Path={filepath}, FourCC=MP4V, FPS={fps:.2f}, Size=({width}x{height})")

            # --- Create Writer ---
            self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

            # --- Check if creation succeeded ---
            if not self.video_writer.isOpened():
                 # If failed, try removing existing file (e.g., 0 byte file) and retry
                 if os.path.exists(filepath):
                      print(f"Warning: VideoWriter failed. Attempting removal: {filepath}")
                      try: os.remove(filepath)
                      except OSError as rm_err: print(f"Could not remove: {rm_err}")
                      # Retry creation after removal attempt
                      self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

                 # If still not opened, raise error
                 if not self.video_writer.isOpened():
                      raise IOError(f"Không thể mở hoặc tạo file video MP4: {filepath}. Codec 'mp4v' có thể không được hỗ trợ.")

            print(f"VideoWriter MP4 created successfully for {filepath}")
            return True # Success

        except Exception as e:
            # Handle any exception during creation
            error_msg = f"Lỗi tạo VideoWriter MP4: {e}"
            QMessageBox.critical(self, "Lỗi Ghi Video", error_msg)
            self._update_status(error_msg)
            print(error_msg, file=sys.stderr)
            # Ensure writer object is cleaned up on error
            if self.video_writer:
                try: self.video_writer.release();
                except: pass
            self.video_writer = None
            return False # Failure


    def _start_recording(self, source="Manual"):
        """Start the video recording process."""
        # --- Pre-checks ---
        # Ensure webcam is running
        if not (self.webcam_thread and self.webcam_thread.isRunning()):
             # Double check needed because of potential race condition?
             # if not self.webcam_thread or not self.webcam_thread.isRunning():
                 QMessageBox.warning(self, "Cảnh báo", "Webcam chưa bật hoặc đang tắt.")
                 self._log_serial(f"[{source}] Ghi thất bại: Webcam không chạy."); return
        # Check if already recording
        if self.is_recording:
             msg = f"Đã {'đang ghi' if not self.is_paused else 'tạm dừng'}."; QMessageBox.warning(self, "Cảnh báo", msg); self._log_serial(f"[{source}] Ghi thất bại: {msg}"); return
        # Check save directory
        if not self.save_directory or not os.path.isdir(self.save_directory):
            QMessageBox.warning(self, "Cảnh báo", "Thư mục lưu không hợp lệ. Vui lòng chọn lại."); self._update_status("Cần chọn thư mục lưu."); return

        # --- Generate Filename & Path ---
        video_filename = self._generate_video_filename()
        full_filepath = os.path.join(self.save_directory, video_filename)

        # --- Create Video Writer ---
        if self._create_video_writer(full_filepath):
            # --- Success: Update State & UI ---
            self.is_recording = True
            self.is_paused = False
            status_msg = f"Bắt đầu ghi: {video_filename}"
            self._update_status(status_msg)
            self._log_serial(f"Bắt đầu ghi [{source}]: {video_filename}")

            # Update buttons
            self.btn_start_record.setEnabled(False) # Cannot start again
            self.btn_pause_record.setEnabled(True) # Can pause
            self.btn_pause_record.setText("Tạm dừng") # Set initial text
            self.btn_stop_save_record.setEnabled(True) # Can stop
            self._update_status_visuals() # Update status label immediately
            # Timer should already be running if properties were received
            if not self.status_timer.isActive(): self.status_timer.start(500)
        else:
             # --- Failure: Clear filename ---
             self.last_video_filename = ""


    def _pause_recording(self, source="Manual"):
        """Toggle pause/resume state of recording."""
        if not self.is_recording:
             self._log_serial(f"[{source}] Pause/Resume bị bỏ qua: Chưa ghi."); return

        self.is_paused = not self.is_paused # Toggle state
        if self.is_paused:
            self.btn_pause_record.setText("Tiếp tục") # Change button text
            status_msg = "Đã tạm dừng ghi video."; log_msg = f"Tạm dừng ghi [{source}]."
        else:
            self.btn_pause_record.setText("Tạm dừng"); status_msg = "Đã tiếp tục ghi video."; log_msg = f"Tiếp tục ghi [{source}]."

        self._update_status(status_msg); self._log_serial(log_msg); self._update_status_visuals()


    def _stop_recording_base(self, action_type, source):
        """Core logic to stop recording (Save or Discard). Returns True if stopped, False otherwise."""
        if not self.is_recording:
             self._log_serial(f"[{source}] Dừng ({action_type}) bị bỏ qua: Chưa ghi."); return False

        print(f"Stop recording ({action_type} by {source}) requested for {self.last_video_filename}")

        self.is_recording = False
        self.is_paused = False

        filepath_to_process = ""
        if self.last_video_filename and self.save_directory:
            filepath_to_process = os.path.join(self.save_directory, self.last_video_filename)
        log_filename = os.path.basename(self.last_video_filename) if self.last_video_filename else "N/A"

        # --- Xử lý VideoWriter: Luôn giải phóng để đảm bảo file được đóng ---
        writer = self.video_writer
        writer_was_opened = False
        release_error = None # <<<--- CHÚ THÍCH: Biến để lưu lỗi nếu có khi release
        if writer:
             writer_was_opened = writer.isOpened()
             self.video_writer = None # Xóa tham chiếu đối tượng

             if writer_was_opened:
                # <<<--- CHÚ THÍCH: LUÔN LUÔN gọi release() để đóng file đúng cách ---
                # Mục đích là để hệ điều hành giải phóng file handle trước khi xóa (nếu là Discard)
                print(f"Releasing writer explicitly to close file handle (Action: {action_type}): {log_filename}")
                try:
                    writer.release()
                    print(f"VideoWriter ({log_filename}) released successfully.")
                except Exception as e:
                    release_error = e # Lưu lỗi lại để xử lý sau
                    print(f"Error releasing writer ({log_filename}): {e}", file=sys.stderr)
             else:
                 print(f"Warning: VideoWriter ({log_filename}) was already closed or not properly opened when stop was requested.")
        else:
            print("Warning: No video writer object found during stop.")

        # <<<--- CHÚ THÍCH: Tạm dừng ngắn (tùy chọn) để OS có thêm thời gian ---
        # Đôi khi sau release(), OS vẫn cần một khoảnh khắc cực nhỏ.
        # Bạn có thể thử bỏ dòng này đi xem có còn lỗi không.
        if action_type == "Discard" and writer_was_opened:
            time.sleep(0.05) # Đợi 50ms (thử nghiệm giá trị này)

        # --- Xóa File nếu hành động là Discard ---
        file_deleted = False
        delete_error = None # <<<--- CHÚ THÍCH: Biến để lưu lỗi khi xóa
        if action_type == "Discard": # Chỉ xóa khi là Discard
            if release_error:
                # <<<--- CHÚ THÍCH: Không thử xóa nếu release đã bị lỗi ---
                print(f"Skipping deletion of {log_filename} because release failed: {release_error}")
                QMessageBox.warning(self, "Lỗi Hủy File", f"Không thể đóng file video đúng cách, không thể xóa:\n{filepath_to_process}\nLỗi release: {release_error}")
            elif filepath_to_process and writer_was_opened: # Chỉ xóa nếu release không lỗi, có đường dẫn, và writer đã mở
                print(f"Attempting to delete discarded file: {filepath_to_process}")
                if os.path.exists(filepath_to_process):
                    try:
                        os.remove(filepath_to_process)
                        file_deleted = True
                        print(f"Successfully deleted discarded file: {log_filename}")
                    except OSError as e:
                        delete_error = e # Lưu lỗi xóa lại
                        print(f"Error deleting discarded file {log_filename}: {e}", file=sys.stderr)
                        QMessageBox.warning(self, "Lỗi Hủy File", f"Không thể xóa file video bị hủy:\n{filepath_to_process}\nLỗi: {e}")
                else:
                    print(f"Discarded file {log_filename} not found (already gone or release failed to create it fully?).")
            elif not writer_was_opened:
                 print(f"Skipping deletion for {log_filename} because the writer wasn't open.")
            else: # Không có filepath_to_process
                 print(f"Skipping deletion because filepath is invalid for {log_filename}.")


        # --- Cập nhật UI & Log dựa trên hành động và kết quả ---
        if action_type == "Save":
             # <<<--- CHÚ THÍCH: Kiểm tra cả lỗi release khi quyết định Save thành công ---
             if filepath_to_process:
                 # Thành công nếu: writer mở, release KHÔNG lỗi, VÀ file tồn tại
                 if writer_was_opened and not release_error and os.path.exists(filepath_to_process):
                     status_msg = f"Đã dừng & lưu: {log_filename}"
                     log_msg = f"Dừng & Lưu [{source}]: {log_filename}"
                     print(f"Video saved successfully confirmed: {filepath_to_process}")
                     self.last_video_filename = "" # Xóa tên file khi lưu thành công
                 else:
                     status_msg = f"LỖI LƯU: {log_filename}"
                     log_msg = f"Dừng & Lỗi Lưu [{source}]: {log_filename}"
                     err_detail = f"Lỗi khi release: {release_error}" if release_error else ("Không tìm thấy file sau khi lưu." if writer_was_opened else "Trình ghi video không mở/lỗi.")
                     QMessageBox.warning(self, "Lưu Thất Bại", f"{err_detail}\n{filepath_to_process}")
             else:
                  status_msg = "Đã dừng (Lỗi tên file/thư mục?)"
                  log_msg = f"Dừng & Lưu [{source}]: Lỗi tên file/thư mục?"
                  QMessageBox.warning(self, "Lưu Thất Bại", f"Không thể lưu: Tên file hoặc thư mục không hợp lệ.")

        else: # action_type == "Discard"
             # <<<--- CHÚ THÍCH: Cập nhật thông báo Discard dựa trên lỗi release và delete ---
             if release_error:
                  delete_status = f"(Lỗi đóng file: {release_error})"
             elif delete_error:
                  delete_status = f"(Lỗi xóa file: {delete_error})"
             elif file_deleted:
                  delete_status = "(Đã xóa file)"
             elif writer_was_opened and filepath_to_process and not os.path.exists(filepath_to_process): # Release ok, xóa ok nhưng file k thấy
                  delete_status = "(Không tìm thấy file sau khi xử lý)"
             elif not writer_was_opened:
                  delete_status = "(Writer không mở)"
             else:
                  delete_status = "(Không thể xác định trạng thái xóa)"

             status_msg = f"Đã dừng & hủy: {log_filename} {delete_status}"
             log_msg = f"Dừng & Hủy [{source}]: {log_filename} {delete_status}"
             self.last_video_filename = "" # Luôn xóa tên file khi hủy

        self._update_status(status_msg); self._log_serial(log_msg)

        # --- Reset Các Nút Điều Khiển Ghi Hình ---
        webcam_can_run = bool(self.webcam_thread and self.webcam_thread.isRunning())
        self.btn_start_record.setEnabled(webcam_can_run)
        self.btn_pause_record.setEnabled(False); self.btn_pause_record.setText("Tạm dừng")
        self.btn_stop_save_record.setEnabled(False)
        self._update_status_visuals()

        return True

    def _stop_save_recording(self, source="Manual"):
        """Stop recording and save the file."""
        # Directly calls base without confirmation for 'Manual' and 'Serial' source
        return self._stop_recording_base("Save", source)

    def _stop_discard_recording(self, source="Manual"):
        """Stop recording and discard the file."""
        # Directly calls base without confirmation
        return self._stop_recording_base("Discard", source)

    def _manual_start_recording(self): self._start_recording("Manual")
    def _manual_pause_recording(self): self._pause_recording("Manual")
    def _manual_stop_save_recording(self):
        # Manual stop save *does not* show confirmation (consistent with Serial command)
        # If confirmation is desired for the button, change this to call _confirm_...
        self._stop_save_recording("Manual")


    # ================== Application Level Methods ==================

    def _confirm_and_stop_recording(self, message):
         """Ask user whether to Save/Discard/Cancel before stopping. Returns True if stopped, False if Cancelled."""
         if not self.is_recording: return True # No recording active, proceed

         reply = QMessageBox.question(self, 'Xác nhận Dừng Ghi Hình', message,
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                     QMessageBox.Save) # Default to Save

         if reply == QMessageBox.Save:
             return self._stop_save_recording("Confirm") # Returns True if stop successful
         elif reply == QMessageBox.Discard:
             return self._stop_discard_recording("Confirm") # Returns True if stop successful
         else: # reply == QMessageBox.Cancel
            return False # User cancelled the parent action

    def closeEvent(self, event):
        """Handle window close event: confirm recording stop, stop threads."""
        print("Close event triggered.")
        should_close = True

        # --- Confirm Stop Recording if active ---
        if self.is_recording:
            print("Recording active. Confirming stop/save before closing...")
            # Use a more explicit confirmation message for closing
            should_close = self._confirm_and_stop_recording(
                "Ứng dụng đang đóng.\nBạn có muốn lưu video đang quay không?"
            )

        if not should_close:
            print("Application close cancelled by user (during recording confirmation).")
            event.ignore() # Prevent window from closing
            return

        # --- Proceed with closing ---
        self._update_status("Đang đóng ứng dụng...")
        QApplication.processEvents() # Update UI to show status message
        self.status_timer.stop() # Stop status updates

        # --- Stop Threads Gracefully ---
        print("Stopping worker threads...")
        # Stop webcam thread (stop includes wait)
        if self.webcam_thread and self.webcam_thread.isRunning():
             # Use internal _stop_webcam as it handles disconnects
             # No need for confirm again, should_close handles that.
             print("Requesting webcam stop...")
             self._stop_webcam()

        # Stop serial thread (disconnect includes wait)
        if self.serial_thread and self.serial_thread.isRunning():
             print("Requesting serial disconnect...")
             self._disconnect_serial()

        # Brief pause allow final resource release (optional, depends on thread wait times)
        # time.sleep(0.1)

        # --- Final Video Writer Check (Safety net) ---
        # This should ideally be released by _confirm_and_stop_recording or _stop_webcam
        if self.video_writer:
             print("Final check: Releasing video writer on exit...")
             if self.video_writer.isOpened():
                 try: self.video_writer.release(); print(" -> Released.")
                 except Exception as e: print(f" -> Error releasing writer on exit: {e}")
             self.video_writer = None

        print("Exiting application cleanly.")
        event.accept() # Allow window to close


# =============================================================================
# == Application Entry Point ==
# =============================================================================
if __name__ == '__main__':
    # --- Optional: High DPI Scaling ---
    # Uncomment one of the methods below BEFORE creating QApplication if needed
    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1" # Simpler method
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) # More explicit
    # QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

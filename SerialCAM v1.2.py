# -*- coding: utf-8 -*-
import sys
import cv2
import serial
import serial.tools.list_ports
import time
import os
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QTextEdit,
                             QFileDialog, QGroupBox, QMessageBox, QSizePolicy,
                             QSpacerItem)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QCoreApplication # Added QCoreApplication

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
        if not self.cap or not self.cap.isOpened():
            print(f"Webcam {self.webcam_index}: Failed with MSMF, trying default backend...")
            if self.cap: self.cap.release() # Ensure released before retrying
            self.cap = cv2.VideoCapture(self.webcam_index) # Fallback to default
            if not self.cap or not self.cap.isOpened():
                # Give it a moment, sometimes cameras need a slight delay
                self.msleep(100)
                self.cap = cv2.VideoCapture(self.webcam_index) # Second try default
                if not self.cap or not self.cap.isOpened():
                    self.error.emit(f"Không thể mở webcam {self.webcam_index} với bất kỳ backend nào.")
                    self._is_running = False
                    print(f"WebcamThread {self.webcam_index}: Failed to open with any backend.")
                    return # Exit run method

        # --- Get properties ---
        try:
            self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._fps = self.cap.get(cv2.CAP_PROP_FPS)

            # Validate dimensions
            if self._width <= 0 or self._height <= 0:
                 raise ValueError(f"Invalid dimensions received: {self._width}x{self._height}")

            # Validate FPS (provide a default if clearly invalid)
            if not (0 < self._fps <= 120): # Set reasonable bounds (e.g., max 120 FPS)
                print(f"Warning: Invalid FPS ({self._fps:.2f}) detected for webcam {self.webcam_index}. Using default 30.0")
                self._fps = 30.0 # Sensible default

            self.properties_ready.emit(self._width, self._height, self._fps)
            print(f"Webcam {self.webcam_index} opened successfully ({self._width}x{self._height} @ {self._fps:.2f} FPS)")

        except Exception as prop_err:
             self.error.emit(f"Lỗi đọc thông số webcam {self.webcam_index}: {prop_err}")
             self._is_running = False
             print(f"WebcamThread {self.webcam_index}: Error getting properties: {prop_err}")
             if self.cap: self.cap.release() # Cleanup on error
             return # Exit run

        # --- Capture loop ---
        while self._is_running:
            if not self.cap or not self.cap.isOpened(): # Check connection health
                 if self._is_running: # Only emit if not stopping
                      self.error.emit(f"Mất kết nối với webcam {self.webcam_index} trong khi chạy.")
                      print(f"WebcamThread {self.webcam_index}: Lost connection during run loop.")
                 self._is_running = False
                 break

            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_ready.emit(frame)
                else:
                    # Check again if loop should be running
                    if self._is_running:
                        # Occasional read failures can happen, maybe try a few times?
                        # For simplicity, we'll treat it as an error for now.
                        print(f"WebcamThread {self.webcam_index}: Frame read failed (ret=False).")
                        # Consider adding a small delay and retry here?
                        self.error.emit(f"Lỗi đọc frame từ webcam {self.webcam_index}.")
                        self._is_running = False # Stop on frame read error
                    break # Exit loop on read failure
            except Exception as read_err:
                 if self._is_running:
                    self.error.emit(f"Lỗi bất ngờ khi đọc frame: {read_err}")
                    print(f"WebcamThread {self.webcam_index}: Exception during read: {read_err}")
                 self._is_running = False # Stop on unexpected error
                 break

            # QCoreApplication.processEvents() # Give GUI thread time (if needed)
            self.msleep(max(1, int(1000 / (self._fps * 1.5)))) # Adjust sleep based on FPS to avoid busy-wait, sleep slightly longer than frame interval

        # --- Cleanup ---
        if self.cap and self.cap.isOpened():
            print(f"WebcamThread {self.webcam_index}: Releasing capture...")
            try:
                self.cap.release()
            except Exception as release_err:
                print(f"Error releasing webcam {self.webcam_index} on thread exit: {release_err}")
        self.cap = None # Clear reference
        print(f"WebcamThread {self.webcam_index}: Exiting run loop.")

    def stop(self):
        """Requests the thread to stop."""
        print(f"WebcamThread {self.webcam_index}: Stop requested.")
        self._is_running = False # Signal the loop to stop
        # Don't release cap here, let run() handle it in its cleanup
        # Wait for the thread to finish its current loop iteration and exit
        if self.isRunning(): # Only wait if it was actually running
            if not self.wait(2000): # Wait up to 2 seconds
                 print(f"Warning: Webcam thread {self.webcam_index} did not finish cleanly after 2s. Terminating.")
                 self.terminate() # Force terminate if stuck (last resort)
                 # Attempt release again after forced termination
                 if self.cap and self.cap.isOpened():
                      try:
                          self.cap.release()
                          print(f"Webcam {self.webcam_index} capture released after termination.")
                      except Exception as e:
                          print(f"Error releasing webcam {self.webcam_index} after termination: {e}")
            else:
                 print(f"WebcamThread {self.webcam_index}: Stopped successfully.")
        else:
             print(f"WebcamThread {self.webcam_index}: Was not running when stop was called.")
        # Safety check after stopping
        if self.cap and self.cap.isOpened():
             print(f"Warning: Webcam {self.webcam_index} capture still open after stop sequence. Releasing.")
             try: self.cap.release()
             except Exception as e: print(f"Error in fallback release for webcam {self.webcam_index}: {e}")
        self.cap = None # Ensure reference is cleared

# =============================================================================
# == Serial Worker Thread == (No changes needed here for looping filenames)
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
            self.data_received.emit("SERIAL_CONNECTED") # Signal successful connection internally

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
                        # Don't yield CPU too much if data is flowing
                        self.msleep(5) # Very small delay
                    else:
                         self.msleep(50) # Sleep longer when no data

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

                # QCoreApplication.processEvents() # May help prevent GUI freeze on rapid serial data? Use with caution.


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
            # self.data_received.emit("SERIAL_DISCONNECTED") # Signal disconnect internally before closing
            port_was = self.port # Store for message before potential clear
            if self.serial_connection and self.serial_connection.isOpen():
                try:
                    print(f"SerialThread {port_was}: Closing serial port in finally block...")
                    self.serial_connection.close()
                except Exception as e:
                     print(f"Error closing serial port {port_was} during run cleanup: {e}")
            self.serial_connection = None # Clear reference
            print(f"SerialThread ({port_was}) exiting run loop.")

    def stop(self):
        """Requests the thread to stop."""
        print(f"SerialThread {self.port}: Stop requested.")
        self._is_running = False # Signal loop to stop
        # Proactively close the port, might help readline() unblock if timeout is long
        if self.serial_connection and self.serial_connection.isOpen():
             try:
                  print(f"SerialThread {self.port}: Closing port from stop() call...")
                  self.serial_connection.close()
                  print(f"SerialThread {self.port}: Port closed by stop() call.")
             except Exception as e:
                  print(f"Error closing serial port {self.port} directly in stop(): {e}")
        # Wait for the run() method to finish (up to timeout)
        if self.isRunning():
            if not self.wait(1500): # Wait 1.5 seconds
                print(f"Warning: Serial thread ({self.port}) did not finish cleanly after 1.5s. Terminating.")
                self.terminate() # Force quit
            else:
                print(f"SerialThread ({self.port}) stopped successfully.")
        else:
             print(f"SerialThread ({self.port}) was not running when stop was called.")
        self.serial_connection = None # Ensure cleared


# =============================================================================
# == Main Application Window ==
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Giám sát Webcam và Điều khiển Serial v1.1 ")
        self.setGeometry(100, 100, 950, 750) # Initial size

        # --- State Variables ---
        self.webcam_thread = None
        self.serial_thread = None
        self.video_writer = None
        self.is_recording = False
        self.is_paused = False
        self.save_directory = os.path.join(os.getcwd(), "Recordings") # Default to 'Recordings' subfolder
        self.webcam_properties = {'width': None, 'height': None, 'fps': None}
        self.last_video_filepath = "" # Store full path of video being recorded
        self.loop_count = 1 # **** ADDED: Loop counter ****

        # --- Timers ---
        self.status_timer = QTimer(self) # For flashing recording status
        self.status_timer.timeout.connect(self._update_status_visuals)
        self.recording_flash_state = False

        # --- Constants ---
        self.common_baud_rates = ["9600", "19200", "38400", "57600", "115200", "250000", "4800", "2400"] # Added 250k
        self.default_baud_rate = "9600"

        # --- Ensure default save directory exists ---
        try:
            os.makedirs(self.save_directory, exist_ok=True)
        except OSError as e:
            print(f"Error creating default save directory '{self.save_directory}': {e}")
            self.save_directory = os.getcwd() # Fallback to current directory

        # --- Initialize UI ---
        self._init_ui()

        # --- Initial Scans & UI Updates ---
        self._scan_webcams()
        self._scan_serial_ports()
        self._update_save_dir_label() # Show initial save directory

        print("MainWindow initialized.")
        self._update_status("Sẵn sàng khởi động Webcam.")


    def _init_ui(self):
        """Build the user interface."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # --- 1. Video Display Area ---
        self.video_frame_label = QLabel("Chưa bật Webcam")
        self.video_frame_label.setAlignment(Qt.AlignCenter)
        self.video_frame_label.setFont(QFont("Arial", 16))
        self.video_frame_label.setStyleSheet("border: 1px solid black; background-color: #e0e0e0;") # Lighter grey
        self.video_frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_frame_label.setMinimumSize(640, 480)
        self.main_layout.addWidget(self.video_frame_label, 1)

        # --- 2. Controls Area (Horizontal Layout) ---
        self.controls_area_widget = QWidget()
        self.controls_area_layout = QHBoxLayout(self.controls_area_widget)
        self.main_layout.addWidget(self.controls_area_widget)

        # --- 2a. Column 1: Webcam & Recording Controls ---
        col1_layout = QVBoxLayout()
        self.controls_area_layout.addLayout(col1_layout, 1)

        # Webcam GroupBox
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

        # Recording GroupBox
        record_group = QGroupBox("Điều khiển Ghi hình")
        record_group_layout = QVBoxLayout()
        save_dir_layout = QHBoxLayout()
        self.lbl_save_dir = QLabel("...")
        self.lbl_save_dir.setStyleSheet("font-style: italic; border: 1px solid #ccc; padding: 2px; background-color: white;")
        self.lbl_save_dir.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.btn_select_dir = QPushButton("Chọn Thư mục Lưu")
        save_dir_layout.addWidget(QLabel("Lưu vào:"))
        save_dir_layout.addWidget(self.lbl_save_dir, 1)
        save_dir_layout.addWidget(self.btn_select_dir)

        record_buttons_layout = QHBoxLayout()
        # self.btn_start_record = QPushButton("Bắt đầu Ghi") # Auto start now
        self.btn_pause_record = QPushButton("Tạm dừng / Tiếp tục")
        self.btn_stop_save_record = QPushButton("Dừng & Lưu Loop")
        # Initially disabled, enabled when recording active
        # self.btn_start_record.setEnabled(False); # Start button no longer primary control
        self.btn_pause_record.setEnabled(False);
        self.btn_stop_save_record.setEnabled(False)
        # record_buttons_layout.addWidget(self.btn_start_record) # Removed start button
        record_buttons_layout.addWidget(self.btn_pause_record)
        record_buttons_layout.addWidget(self.btn_stop_save_record)

        self.lbl_record_status = QLabel("Trạng thái: Chờ Webcam")
        self.lbl_record_status.setAlignment(Qt.AlignCenter)
        self.lbl_record_status.setFont(QFont("Arial", 11, QFont.Bold))

        record_group_layout.addLayout(save_dir_layout)
        record_group_layout.addLayout(record_buttons_layout)
        record_group_layout.addWidget(self.lbl_record_status)
        record_group.setLayout(record_group_layout)
        col1_layout.addWidget(record_group)
        col1_layout.addStretch()

        # --- 2b. Column 2: Serial Controls & Log ---
        col2_layout = QVBoxLayout()
        self.controls_area_layout.addLayout(col2_layout, 1)

        # Serial GroupBox
        serial_group = QGroupBox("Điều khiển Cổng Serial (COM)")
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
        serial_log_layout.addWidget(QLabel("Log Điều Khiển & Serial:")) # Renamed label
        self.serial_log = QTextEdit()
        self.serial_log.setReadOnly(True)
        self.serial_log.setFixedHeight(150) # Slightly taller log
        self.serial_log.setFont(QFont("Consolas", 9))
        serial_log_layout.addWidget(self.serial_log)

        serial_layout_main.addLayout(serial_config_layout)
        serial_layout_main.addLayout(serial_connect_layout)
        serial_layout_main.addLayout(serial_log_layout)
        serial_group.setLayout(serial_layout_main)
        col2_layout.addWidget(serial_group)
        col2_layout.addStretch()

        # --- 3. Bottom Area: Exit Button & Status Bar ---
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        self.btn_exit = QPushButton("Thoát")
        bottom_layout.addWidget(self.btn_exit)
        self.main_layout.addLayout(bottom_layout)

        # Status Bar
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
        # Recording Controls
        self.btn_select_dir.clicked.connect(self._select_save_directory)
        # self.btn_start_record.clicked.connect(self._manual_start_recording) # Button removed
        self.btn_pause_record.clicked.connect(self._manual_pause_recording)
        self.btn_stop_save_record.clicked.connect(self._manual_stop_save_recording)
        # Serial Controls
        self.btn_scan_serial.clicked.connect(self._scan_serial_ports)
        self.btn_connect_serial.clicked.connect(self._connect_serial)
        self.btn_disconnect_serial.clicked.connect(self._disconnect_serial)
        # Exit Button
        self.btn_exit.clicked.connect(self.close)


    # ================== UI Update Slots & Helpers ==================

    def _log(self, message, source="System"):
        """Append a timestamped message to the log area."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3] # Added milliseconds
        prefix = f"[{source} {timestamp}]"
        full_message = f"{prefix} {message}"

        self.serial_log.append(full_message)
        self.serial_log.ensureCursorVisible() # Auto-scroll

        # Also print important messages to console for debugging
        # Use source or keywords to decide if console output is needed
        if source in ["System", "SerialCmd", "Webcam", "Record"] or "LỖI" in message or "Error" in message:
             print(full_message)

    def _update_status(self, message):
        """Update the status bar message and print to console for debug."""
        self.status_label.setText(f" {message}")
        # print(f"StatusBar: {message}") # Can be noisy, log() is better

    def _update_status_visuals(self):
        """Update the recording status label (flashing effect)."""
        status_text = "Chờ Webcam"
        style_sheet = "color: black;" # Default style

        if self.webcam_thread and self.webcam_thread.isRunning():
             if self.is_recording:
                  # Determine filename part for display
                  display_filename = os.path.basename(self.last_video_filepath) if self.last_video_filepath else "..."
                  self.recording_flash_state = not self.recording_flash_state # Toggle flash state
                  # Set text and style based on paused state and flash state
                  base_text_recording = f"ĐANG GHI Loop {self.loop_count}: {display_filename}"
                  base_text_paused = f"TẠM DỪNG Loop {self.loop_count}: {display_filename}"
                  if self.is_paused:
                       status_text = base_text_paused
                       style_sheet = "color: orange; font-weight: bold;" # Orange for paused
                  else:
                       status_text = base_text_recording
                       # Red flashing for recording
                       style_sheet = "color: red; font-weight: bold;" if self.recording_flash_state else "color: darkred; font-weight: bold;"
                  self.lbl_record_status.setText(status_text)
             else:
                  # Webcam is on but recording is initializing or stopped between loops
                  status_text = "Webcam Bật (Sẵn sàng ghi)"
                  style_sheet = "color: green; font-weight: bold;"
                  self.lbl_record_status.setText(f"Trạng thái: {status_text}")
        elif self.webcam_thread is None: # Webcam explicitly stopped or not started
            status_text = "Webcam Đã Tắt"
            style_sheet = "color: gray;"
            self.lbl_record_status.setText(f"Trạng thái: {status_text}")
        # else (e.g., thread exists but not running - during shutdown): Keep current text? or indicate stopping?

        self.lbl_record_status.setStyleSheet(style_sheet)


    def _update_save_dir_label(self):
        """Update the save directory label, shortening if needed."""
        display_path = self.save_directory
        max_len = 45 # Adjust max characters to fit UI
        if len(display_path) > max_len:
            # Show beginning and end parts
            parts = display_path.split(os.sep)
            if len(parts) > 2:
                display_path = os.path.join(parts[0], "...", parts[-2], parts[-1]) # e.g. C:\...\Folder\Subfolder
                if len(display_path) > max_len: # If still too long, truncate end
                    display_path = display_path[:max_len-3] + "..."
            else: # Fallback simple truncate if path structure is unusual
                display_path = display_path[:max_len-3] + "..."
        self.lbl_save_dir.setText(display_path)
        self.lbl_save_dir.setToolTip(self.save_directory) # Show full path on hover


    # ================== Webcam Control Methods ==================

    def _scan_webcams(self):
        """Scan for available webcams and update the combobox."""
        # Prevent scanning if webcam is active
        if self.webcam_thread and self.webcam_thread.isRunning():
             QMessageBox.warning(self, "Thông báo", "Vui lòng tắt webcam trước khi quét lại.")
             return

        self.combo_webcam.clear()
        available_webcams = []
        index = 0
        max_scan_index = 5
        self._log("Bắt đầu quét webcam...", "System")
        QApplication.setOverrideCursor(Qt.WaitCursor) # Indicate busy
        self.statusBar.showMessage("Đang quét webcam...")
        while index < max_scan_index:
             cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
             opened = cap is not None and cap.isOpened()
             if not opened:
                 if cap: cap.release()
                 cap = cv2.VideoCapture(index) # Try default
                 opened = cap is not None and cap.isOpened()

             if opened:
                 cam_name = f"Webcam {index}" # Basic name
                 # Could try getting properties here for more info, but slows down scan
                 available_webcams.append((index, cam_name))
                 print(f"  Found active webcam at index: {index}")
                 cap.release() # IMPORTANT: Release after check
                 index += 1
             else:
                 if cap: cap.release()
                 print(f"  No active webcam at index: {index}. Stopping scan.")
                 break # Stop scanning if an index fails

        QApplication.restoreOverrideCursor() # Restore cursor
        self.statusBar.clearMessage()

        if not available_webcams:
            self.combo_webcam.addItem("Không tìm thấy webcam")
            self.btn_start_webcam.setEnabled(False)
            self._update_status("Quét xong: Không tìm thấy webcam.")
            self._log("Quét webcam hoàn tất: Không tìm thấy", "System")
        else:
            for idx, name in available_webcams: self.combo_webcam.addItem(name, userData=idx)
            self.btn_start_webcam.setEnabled(True)
            self._update_status(f"Quét xong: Tìm thấy {len(available_webcams)} webcam.")
            self._log(f"Quét webcam hoàn tất: Tìm thấy {len(available_webcams)} webcam.", "System")
            if len(available_webcams) > 0: self.combo_webcam.setCurrentIndex(0)


    def _start_webcam(self):
        """Start the selected webcam."""
        if self.webcam_thread and self.webcam_thread.isRunning():
            QMessageBox.warning(self, "Thông báo", "Webcam đã đang chạy.")
            return
        selected_index = self.combo_webcam.currentIndex()
        if selected_index < 0 or self.combo_webcam.itemData(selected_index) is None:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn webcam hợp lệ hoặc quét lại.")
            return

        webcam_idx = self.combo_webcam.itemData(selected_index)
        self._log(f"Bắt đầu khởi động webcam {webcam_idx}...", "Webcam")

        self.video_frame_label.setText(f"Đang kết nối Webcam {webcam_idx}...")
        self.video_frame_label.setFont(QFont("Arial", 16))
        self.video_frame_label.setStyleSheet("border: 1px solid black; background-color: #e0e0e0;")
        self.video_frame_label.repaint()
        self.btn_start_webcam.setEnabled(False)
        self.btn_stop_webcam.setEnabled(True) # Enable stop immediately
        self.combo_webcam.setEnabled(False)
        self.btn_scan_webcam.setEnabled(False)
        self._update_status(f"Đang khởi động Webcam {webcam_idx}...")
        # Reset recording buttons
        # self.btn_start_record.setEnabled(False) # Start button no longer used
        self.btn_pause_record.setEnabled(False)
        self.btn_stop_save_record.setEnabled(False)
        self._update_status_visuals() # Update status label to "Chờ Webcam" or similar

        # --- Create and start thread ---
        self.webcam_thread = WebcamThread(webcam_idx)
        self.webcam_thread.frame_ready.connect(self._update_frame)
        self.webcam_thread.error.connect(self._handle_webcam_error)
        self.webcam_thread.properties_ready.connect(self._on_webcam_properties_ready)
        self.webcam_thread.finished.connect(self._on_webcam_thread_finished)
        self.webcam_thread.start()


    def _on_webcam_properties_ready(self, width, height, fps):
        """Slot called when webcam properties are successfully retrieved. Initiates auto-recording."""
        # Ensure this slot is responding to the correct, current thread
        if self.webcam_thread and self.sender() == self.webcam_thread and self.webcam_thread.isRunning():
            self.webcam_properties = {'width': width, 'height': height, 'fps': fps}
            status_msg = f"Webcam {self.combo_webcam.currentData()} bật [{width}x{height} @ {fps:.2f} FPS]."
            self._update_status(status_msg)
            self._log(status_msg, "Webcam")
            # Automatically start the first recording loop
            self._log("Webcam sẵn sàng, tự động bắt đầu ghi loop 1...", "Record")
            # Ensure status visuals reflect "Webcam On" before starting record
            self._update_status_visuals()
            QApplication.processEvents() # Allow UI to update
            self._start_recording("Auto") # Call auto-start
            # Timer for status flashing is started inside _start_recording if successful
        else:
            print("Warning: _on_webcam_properties_ready called for non-active/mismatched thread.")
            self._log("Nhận được thông số webcam nhưng thread không hợp lệ.", "System")


    def _stop_webcam(self):
        """Stop the running webcam thread and handle recording state."""
        if not (self.webcam_thread and self.webcam_thread.isRunning()):
             print("Stop webcam request ignored: No webcam running.")
             self._log("Yêu cầu tắt webcam bị bỏ qua (không có webcam đang chạy)", "System")
             # Ensure UI is in stopped state if thread object exists but isn't running
             if self.webcam_thread and not self.webcam_thread.isRunning():
                  self._on_webcam_thread_finished() # Force UI cleanup
             return

        self._log("Yêu cầu dừng webcam...", "Webcam")
        self.status_timer.stop() # Stop visual updates first

        # --- Handle ongoing recording before stopping thread ---
        should_proceed_with_stop = True
        if self.is_recording:
            self._log("Webcam đang tắt nhưng có ghi hình đang chạy. Xác nhận Lưu/Hủy...", "Record")
            should_proceed_with_stop = self._confirm_and_stop_recording(
                f"Webcam sắp tắt.\nVideo hiện tại ({os.path.basename(self.last_video_filepath or '...')}) chưa được lưu.\nBạn có muốn lưu lại Loop {self.loop_count} không?"
            )
            # _confirm_and_stop_recording handles the stop/save/discard AND increments counter.
            # We specifically DON'T want it to auto-restart in this scenario.
            # The _stop_recording_base logic should prevent restart if webcam thread isn't running.

        if not should_proceed_with_stop:
             self._log("Việc tắt webcam bị hủy bởi người dùng.", "System")
             # If recording was active and user cancelled, restart timer
             if self.is_recording:
                 self.status_timer.start(500)
             return # Abort webcam stop process

        # --- Proceed with stopping the thread ---
        self._log("Tiến hành dừng thread webcam...", "Webcam")
        self._update_status("Đang tắt webcam...")
        # Indicate stopping on video label
        self.video_frame_label.setText("Đang tắt Webcam...")
        self.video_frame_label.setFont(QFont("Arial", 16))
        self.video_frame_label.setStyleSheet("border: 1px solid black; background-color: #e0e0e0;")
        self.video_frame_label.repaint()
        # Prevent starting again while stopping
        self.btn_start_webcam.setEnabled(False)
        self.combo_webcam.setEnabled(False)
        self.btn_scan_webcam.setEnabled(False)
        # Disable recording controls during stop
        self.btn_pause_record.setEnabled(False);
        self.btn_stop_save_record.setEnabled(False)

        thread_to_stop = self.webcam_thread
        if thread_to_stop:
             # Disconnect signals *before* calling stop
             signals_to_disconnect = [
                 (thread_to_stop.frame_ready, self._update_frame),
                 (thread_to_stop.error, self._handle_webcam_error),
                 (thread_to_stop.properties_ready, self._on_webcam_properties_ready),
                 (thread_to_stop.finished, self._on_webcam_thread_finished)
             ]
             for signal, slot in signals_to_disconnect:
                 try: signal.disconnect(slot)
                 except: pass # Ignore if already disconnected

             thread_to_stop.stop() # Request thread stop (includes wait)
             # _on_webcam_thread_finished (connected to finished signal) will handle final UI reset AFTER thread truly exits


    def _on_webcam_thread_finished(self):
        """Slot called when the WebcamThread has completely finished."""
        self._log("Thread webcam đã kết thúc.", "Webcam")
        self.webcam_thread = None # Crucial: Clear the reference

        # Reset video display
        self.video_frame_label.setText("Webcam Đã Tắt")
        self.video_frame_label.setFont(QFont("Arial", 16))
        self.video_frame_label.setStyleSheet("border: 1px solid black; background-color: #e0e0e0;")
        self.video_frame_label.setPixmap(QPixmap()) # Clear image

        # Reset webcam controls
        self.btn_start_webcam.setEnabled(True)
        self.btn_stop_webcam.setEnabled(False)
        self.combo_webcam.setEnabled(True)
        self.btn_scan_webcam.setEnabled(True)

        # Reset recording state variables (should have been done by stop sequence, but ensure here)
        self.is_recording = False
        self.is_paused = False
        self.webcam_properties = {'width': None, 'height': None, 'fps': None} # Clear properties
        self.last_video_filepath = "" # Clear last filename path

        # **** Reset loop count when webcam is stopped? ****
        # Decision: Keep the loop count incrementing across webcam stop/starts for simplicity now.
        # To reset on each webcam start, uncomment the next line and potentially adjust _start_webcam.
        # self.loop_count = 1

        # Reset recording buttons
        # self.btn_start_record.setEnabled(False)
        self.btn_pause_record.setEnabled(False); self.btn_pause_record.setText("Tạm dừng / Tiếp tục")
        self.btn_stop_save_record.setEnabled(False)

        # Final cleanup of video writer (safety net)
        if self.video_writer:
            self._log("Cảnh báo: Phát hiện video writer chưa đóng khi webcam kết thúc. Đang đóng...", "System")
            if self.video_writer.isOpened():
                 try: self.video_writer.release()
                 except Exception as e: self._log(f"Lỗi đóng writer tồn đọng: {e}", "System")
            self.video_writer = None

        # Ensure status timer is stopped and update UI
        self.status_timer.stop()
        self._update_status("Webcam đã tắt.")
        self._update_status_visuals() # Update recording status label


    def _handle_webcam_error(self, message):
        """Handle errors emitted by the webcam thread."""
        # Only process error if it's from the currently active thread
        if self.webcam_thread and self.sender() == self.webcam_thread:
             error_prefix = "LỖI WEBCAM:"
             self._log(f"{error_prefix} {message}", "Webcam")
             self._update_status(f"{error_prefix} Xem Log.")
             QMessageBox.critical(self, "Lỗi Webcam", message)
             # Webcam errored, try to stop everything gracefully
             # The webcam thread might already be stopping itself upon error emission
             if self.webcam_thread.isRunning():
                 self._stop_webcam() # Initiate the stop sequence (will prompt save if needed)
             else:
                  # If thread already stopped, just ensure UI cleanup
                  self._on_webcam_thread_finished()
        else:
             # This can happen if an error signal arrives after self.webcam_thread has been reset (e.g., during stop)
             print(f"Ignoring delayed/mismatched webcam error signal: {message}")


    def _update_frame(self, frame):
        """Update the video display label and write frame to video if recording."""
        if frame is None: return

        current_time = time.monotonic() # For timing frame processing if needed

        # --- Write frame to video FIRST if recording ---
        # Prioritize writing over displaying to minimize frame drops if UI is slow
        writer = self.video_writer # Local reference for safety
        write_error = False
        if self.is_recording and not self.is_paused and writer and writer.isOpened():
             try:
                 writer.write(frame) # Write the original BGR frame
             except Exception as e:
                 # Handle write error
                 error_msg = f"Lỗi ghi frame video: {e}"
                 self._log(error_msg, "Record")
                 print(error_msg, file=sys.stderr)
                 write_error = True
                 # Auto-stop recording on write error to prevent flooding with errors
                 self._stop_recording_base("Discard", "WriteError") # Discard the corrupted file

        # --- Display Frame ---
        # Don't display if there was a critical write error that stopped recording
        if not write_error and self.video_frame_label.isVisible(): # Check visibility
             try:
                  # Conversion and Scaling - keep this efficient
                  label_width = self.video_frame_label.width()
                  label_height = self.video_frame_label.height()
                  h, w, ch = frame.shape

                  # Avoid unnecessary conversion/scaling if dimensions match? Probably negligible gain.
                  if w > 0 and h > 0 and label_width > 0 and label_height > 0:
                      rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                      bytes_per_line = ch * w
                      qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                      pixmap = QPixmap.fromImage(qt_image)
                      # Keep aspect ratio, use fast scaling
                      scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.FastTransformation)
                      self.video_frame_label.setPixmap(scaled_pixmap)
                  # else: keep previous frame or show placeholder? For now, do nothing if bad dimensions

             except Exception as e:
                 # Use time.time() based throttling if errors are continuous
                 print(f"Error converting/displaying frame: {e}")

        # Process UI events occasionally to keep responsive, but not too often
        # QCoreApplication.processEvents()


    # ================== Serial Control Methods ==================

    def _scan_serial_ports(self):
        """Scan for available serial ports and update the combobox."""
        if self.serial_thread and self.serial_thread.isRunning():
             QMessageBox.warning(self, "Thông báo", "Vui lòng ngắt kết nối Serial trước khi quét lại.")
             return

        self.combo_com_port.clear()
        self._log("Bắt đầu quét cổng Serial...", "Serial")
        QApplication.setOverrideCursor(Qt.WaitCursor) # Indicate busy
        self.statusBar.showMessage("Đang quét cổng COM...")
        ports = serial.tools.list_ports.comports()
        found_ports = []

        if ports:
             for port in sorted(ports, key=lambda p: p.device):
                  # Filter common device names (adjust as needed for OS/device)
                  dev_upper = port.device.upper()
                  if "COM" in dev_upper or "ACM" in dev_upper or "USB" in dev_upper or "TTY" in dev_upper:
                      desc = f" - {port.description}" if port.description and port.description != "n/a" else ""
                      # Optional: Try opening the port briefly to see if it's usable? (Can be slow/risky)
                      found_ports.append((port.device, f"{port.device}{desc}"))
                      print(f"  Found potential port: {port.device}{desc}")
        else:
             print("  No serial ports reported by system.")


        QApplication.restoreOverrideCursor() # Restore cursor
        self.statusBar.clearMessage()

        if not found_ports:
            self.combo_com_port.addItem("Không tìm thấy cổng COM")
            self.btn_connect_serial.setEnabled(False)
            self._update_status("Quét xong: Không tìm thấy cổng COM.")
            self._log("Quét Serial hoàn tất: Không tìm thấy.", "Serial")
        else:
            for device, name in found_ports: self.combo_com_port.addItem(name, userData=device)
            self.btn_connect_serial.setEnabled(True)
            self._update_status(f"Quét xong: Tìm thấy {len(found_ports)} cổng COM.")
            self._log(f"Quét Serial hoàn tất: Tìm thấy {len(found_ports)}.", "Serial")
            if len(found_ports) > 0: self.combo_com_port.setCurrentIndex(0) # Select first one


    def _connect_serial(self):
        """Connect to the selected serial port with the selected baud rate."""
        if self.serial_thread and self.serial_thread.isRunning():
            QMessageBox.warning(self, "Thông báo", "Đã kết nối Serial.")
            return

        selected_port_index = self.combo_com_port.currentIndex()
        if selected_port_index < 0 or self.combo_com_port.itemData(selected_port_index) is None:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn cổng COM hợp lệ hoặc quét lại.")
            return
        port_name = self.combo_com_port.itemData(selected_port_index)

        selected_baud_text = self.combo_baud_rate.currentText()
        try:
            baud_rate = int(selected_baud_text)
            if baud_rate <= 0: raise ValueError("Baud rate phải dương.")
        except ValueError:
            QMessageBox.critical(self, "Lỗi Baudrate", f"Baudrate không hợp lệ: '{selected_baud_text}'.")
            return

        self._log(f"Đang kết nối tới Serial: {port_name} @ {baud_rate} baud...", "Serial")
        self._update_status(f"Đang kết nối {port_name}...")
        self.serial_log.repaint()
        self.btn_connect_serial.setEnabled(False)
        # Don't enable disconnect immediately, wait for confirmation from thread
        self.btn_disconnect_serial.setEnabled(False)
        self.combo_com_port.setEnabled(False)
        self.combo_baud_rate.setEnabled(False)
        self.btn_scan_serial.setEnabled(False)

        # --- Create and start thread ---
        self.serial_thread = SerialThread(port_name, baudrate=baud_rate)
        # Use a lambda to pass additional args if needed, or check data internally
        self.serial_thread.data_received.connect(self._handle_serial_data)
        self.serial_thread.error.connect(self._handle_serial_error)
        self.serial_thread.finished.connect(self._on_serial_thread_finished)
        self.serial_thread.start()


    def _disconnect_serial(self):
        """Disconnect the currently active serial connection."""
        if self.serial_thread and self.serial_thread.isRunning():
             port = self.serial_thread.port # Get info before stopping
             baud = self.serial_thread.baudrate
             self._log(f"Đang ngắt kết nối Serial ({port}@{baud})...", "Serial")
             self._update_status(f"Đang ngắt kết nối Serial ({port})...")

             # Prevent new connection attempts while disconnecting
             self.btn_connect_serial.setEnabled(False)
             self.btn_disconnect_serial.setEnabled(False)

             thread_to_stop = self.serial_thread
             if thread_to_stop:
                 # Disconnect signals first
                 signals_to_disconnect = [
                    (thread_to_stop.data_received, self._handle_serial_data),
                    (thread_to_stop.error, self._handle_serial_error),
                    (thread_to_stop.finished, self._on_serial_thread_finished)
                 ]
                 for signal, slot in signals_to_disconnect:
                     try: signal.disconnect(slot)
                     except: pass

                 # Request thread stop (includes proactive close() and wait())
                 thread_to_stop.stop()
                 # _on_serial_thread_finished (connected to finished) handles the final UI state reset
             else: # Should not happen if isRunning was true, but as safety:
                 self._on_serial_thread_finished() # Reset UI manually if ref lost
        else:
             self._log("Yêu cầu ngắt kết nối Serial bị bỏ qua (không có kết nối).", "System")
             # Ensure UI is in disconnected state if no thread object exists
             if not self.serial_thread: self._on_serial_thread_finished()


    def _on_serial_thread_finished(self):
        """Slot called when the SerialThread has completely finished."""
        # Check if the thread object still exists before logging disconnect.
        # This avoids logging "Disconnected" if connect failed initially.
        if hasattr(self, 'serial_thread_just_finished') and self.serial_thread_just_finished:
             # Log disconnect only if we know it was connected
             self._log("Đã ngắt kết nối Serial.", "Serial")
             self._update_status("Đã ngắt kết nối Serial.")

        self._log("Thread Serial đã kết thúc.", "Serial")
        self.serial_thread = None # Clear reference
        self.serial_thread_just_finished = False # Reset flag

        # Reset UI elements to disconnected state
        self.btn_connect_serial.setEnabled(True) # Allow attempting connection again
        self.btn_disconnect_serial.setEnabled(False) # Disconnect impossible now
        self.combo_com_port.setEnabled(True)
        self.combo_baud_rate.setEnabled(True)
        self.btn_scan_serial.setEnabled(True)
        # Status bar might show webcam status if running, or ready if not


    def _handle_serial_error(self, message):
        """Handle errors emitted by the serial thread."""
        if self.serial_thread and self.sender() == self.serial_thread:
            error_prefix = "LỖI SERIAL:"
            self._log(f"{error_prefix} {message}", "Serial")
            self._update_status(f"{error_prefix} Xem Log.")
            print(f"Serial Error: {message}", file=sys.stderr)
            QMessageBox.critical(self, "Lỗi Serial", message)
            # The serial thread likely stops itself on error.
            # We just need to wait for the 'finished' signal to reset the UI.
            # Proactively disable disconnect button if error occurs while connected
            self.btn_disconnect_serial.setEnabled(False)
            # Mark that the thread finished because of an error (handled in _on_serial_thread_finished)
            self.serial_thread_just_finished = True
        else:
             # May happen if error arrives after thread stopped/disconnected
             print(f"Ignoring delayed/mismatched serial error signal: {message}")


    def _handle_serial_data(self, data):
        """Process data/commands received from the serial port."""
        # Special internal signal from thread on successful connection
        if data == "SERIAL_CONNECTED":
             self._log("Kết nối Serial thành công.", "Serial")
             self._update_status(f"Đã kết nối {self.serial_thread.port}@{self.serial_thread.baudrate}.")
             # Enable disconnect button ONLY after successful connect
             self.btn_disconnect_serial.setEnabled(True)
             self.serial_thread_just_finished = True # Mark as was connected for disconnect logging later
             return
        # Internal signal for disconnect could be added too if needed

        self._log(f"Nhận Serial: '{data}'", "Serial")
        command = data.strip().upper() # Normalize command

        # Check if webcam is running for commands that need it
        webcam_running = bool(self.webcam_thread and self.webcam_thread.isRunning())

        # --- Process known commands ---
        # Commands that ALWAYS work:
        if command == "PING":
             self._log("Pong!", "Serial") # Example simple reply
             # Can send back via self.serial_thread.serial_connection.write(b'PONG\n') if needed

        # Commands that require WEBCAM:
        elif command == "START": # Less useful now, maybe acts as RESUME?
             if webcam_running:
                 if self.is_recording and self.is_paused:
                     self._pause_recording("SerialCmd") # Treat as Resume
                 elif self.is_recording:
                     self._log("Lệnh 'START' bị bỏ qua: Đang ghi.", "SerialCmd")
                 else: # Not recording (maybe between loops?)
                      self._log("Lệnh 'START' bị bỏ qua: Ghi tự động khi webcam bật.", "SerialCmd")
             else: self._log("Lệnh 'START' bị bỏ qua: Webcam chưa bật.", "SerialCmd")

        elif command == "STOP_SAVE":
            if webcam_running and self.is_recording:
                self._stop_save_recording("SerialCmd") # This will increment and auto-restart
            else: self._log("Lệnh 'STOP_SAVE' bị bỏ qua: Chưa bật webcam hoặc chưa ghi.", "SerialCmd")

        elif command == "STOP_DISCARD":
            if webcam_running and self.is_recording:
                self._stop_discard_recording("SerialCmd") # This will increment and auto-restart
            else: self._log("Lệnh 'STOP_DISCARD' bị bỏ qua: Chưa bật webcam hoặc chưa ghi.", "SerialCmd")

        elif command == "PAUSE":
            if webcam_running and self.is_recording and not self.is_paused:
                 self._pause_recording("SerialCmd")
            else: self._log("Lệnh 'PAUSE' bị bỏ qua: Chưa ghi hoặc đã dừng.", "SerialCmd")

        elif command == "RESUME":
            if webcam_running and self.is_recording and self.is_paused:
                 self._pause_recording("SerialCmd") # Same action toggles pause state
            else: self._log("Lệnh 'RESUME' bị bỏ qua: Chưa ghi hoặc đang chạy.", "SerialCmd")

        else:
            self._log(f"Lệnh không xác định từ Serial: '{command}' (Gốc: '{data}')", "Serial")


    # ================== Recording Control Methods ==================

    def _select_save_directory(self):
        """Open dialog to select video save directory."""
        new_dir = QFileDialog.getExistingDirectory(self, "Chọn Thư mục Lưu Video Mới", self.save_directory)
        if new_dir and new_dir != self.save_directory:
            # Check if writable? Simple check: try creating a temp file/dir
            temp_test_file = os.path.join(new_dir, f"~test_write_{int(time.time())}.tmp")
            can_write = False
            try:
                os.makedirs(os.path.dirname(temp_test_file), exist_ok=True) # Ensure parent dir exists
                with open(temp_test_file, "w") as f: f.write("test")
                os.remove(temp_test_file)
                can_write = True
                self.save_directory = new_dir
                self._update_save_dir_label()
                self._update_status(f"Thư mục lưu được đặt thành: {self.save_directory}")
                self._log(f"Thư mục lưu được đổi thành: {self.save_directory}", "System")
            except Exception as e:
                 self._log(f"Lỗi khi kiểm tra quyền ghi vào thư mục '{new_dir}': {e}", "System")
                 QMessageBox.warning(self, "Lỗi Thư mục", f"Không thể ghi vào thư mục đã chọn:\n{new_dir}\n\nLỗi: {e}\n\nVui lòng chọn thư mục khác.")
                 self.save_directory = os.path.join(os.getcwd(), "Recordings") # Revert to default on error
                 os.makedirs(self.save_directory, exist_ok=True)
                 self._update_save_dir_label()

        elif new_dir == self.save_directory:
            self._log("Đã chọn cùng thư mục lưu, không thay đổi.", "System")
        else: # Dialog cancelled
             self._update_status("Việc chọn thư mục bị hủy.")


    def _generate_video_filename(self):
        """Generate a filename using the loop count and specific timestamp format."""
        now = datetime.now()
        # Format: loop<count>_ddmmyyyy_HHMMSS.mp4
        filename = now.strftime(f"loop{self.loop_count}_%d%m%Y_%H%M%S.mp4")
        # Store the full path for potential use (e.g., logging, cleanup)
        full_path = os.path.join(self.save_directory, filename)
        self.last_video_filepath = full_path # Store full path being recorded
        self._log(f"Tạo tên file mới: {filename} (Loop {self.loop_count})", "Record")
        return full_path


    def _create_video_writer(self, filepath):
        """Initialize the OpenCV VideoWriter for MP4."""
        props = self.webcam_properties
        if not all(props.values()) or props['width'] <= 0 or props['height'] <= 0 or props['fps'] <= 0:
             error_msg = f"Lỗi: Thông số webcam không hợp lệ khi tạo writer: {props}"
             self._log(error_msg, "Record")
             QMessageBox.critical(self, "Lỗi Ghi Video", f"{error_msg}\nKhông thể bắt đầu ghi.")
             self._update_status(error_msg)
             return False # Indicate failure

        # Define Codec (MP4 - try alternatives if mp4v fails)
        fourcc_options = [cv2.VideoWriter_fourcc(*'mp4v'), # Preferred
                          cv2.VideoWriter_fourcc(*'avc1'), # H.264
                          cv2.VideoWriter_fourcc(*'XVID'), # AVI container often uses this
                          0x00000021 # Fallback MJPG on some Linux (check if available)
                         ]
        codec_name = "'mp4v'" # Default for messages
        width = props['width']; height = props['height']; fps = props['fps']
        # Re-validate FPS again just before creation
        if not (1 <= fps <= 120): # Set stricter bounds (min 1 fps)
             print(f"Warning: FPS ({fps}) ngoài phạm vi hợp lệ (1-120) khi tạo writer. Sử dụng 30.0")
             fps = 30.0

        self.video_writer = None # Ensure it's None initially
        success = False
        for fourcc in fourcc_options:
             if fourcc == cv2.VideoWriter_fourcc(*'avc1'): codec_name = "'avc1 (H.264)'"
             elif fourcc == cv2.VideoWriter_fourcc(*'XVID'): codec_name = "'XVID'"
             elif fourcc == 0x00000021 : codec_name = "'MJPG'" # Used for Linux often

             try:
                 # Ensure target directory exists
                 os.makedirs(os.path.dirname(filepath), exist_ok=True)
                 self._log(f"Thử tạo VideoWriter: Path={os.path.basename(filepath)}, Codec={codec_name}, FPS={fps:.2f}, Size=({width}x{height})", "Record")

                 self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

                 if self.video_writer and self.video_writer.isOpened():
                     self._log(f"VideoWriter MP4 ({codec_name}) tạo thành công cho {os.path.basename(filepath)}", "Record")
                     success = True
                     break # Exit loop on success
                 else:
                      self._log(f"Thất bại với codec {codec_name}.", "Record")
                      # Clean up writer instance if creation failed but object exists
                      if self.video_writer: self.video_writer = None
                      # Attempt to remove potentially empty/corrupt file created by failed attempt
                      if os.path.exists(filepath):
                            try: os.remove(filepath)
                            except OSError: pass # Ignore if remove fails

             except Exception as e:
                  self._log(f"Lỗi khi thử tạo VideoWriter với codec {codec_name}: {e}", "Record")
                  if self.video_writer: self.video_writer = None # Cleanup instance
                  if os.path.exists(filepath):
                       try: os.remove(filepath)
                       except OSError: pass
                  continue # Try next codec

        if not success:
            error_msg = f"LỖI: Không thể tạo file video MP4: {os.path.basename(filepath)}. Không có codec phù hợp (thử mp4v, avc1, XVID) hoặc lỗi ghi."
            self._log(error_msg, "Record")
            QMessageBox.critical(self, "Lỗi Ghi Video", error_msg)
            self._update_status(error_msg)
            self.video_writer = None # Ensure it's None
            return False # Indicate failure

        return True # Indicate success


    def _start_recording(self, source="Manual"):
        """Start or restart a recording loop."""
        # --- Pre-checks ---
        # Ensure webcam is running AND has valid properties
        if not (self.webcam_thread and self.webcam_thread.isRunning() and all(self.webcam_properties.values())):
                 msg = "Webcam chưa sẵn sàng hoặc thông số không hợp lệ."
                 self._log(f"[{source}] Ghi thất bại: {msg}", "Record")
                 if source != "Auto-Restart": # Don't show popup during rapid restart
                    QMessageBox.warning(self, "Cảnh báo Ghi Hình", msg)
                 return False # Indicate start failed

        # If called while already recording (shouldn't happen in auto mode unless error/race condition)
        if self.is_recording:
             msg = f"Yêu cầu ghi [{source}] bị bỏ qua: Đã {'đang ghi' if not self.is_paused else 'tạm dừng'} loop {self.loop_count}."
             self._log(msg, "Record")
             # No popup needed here, just log it.
             return False

        # Check save directory one last time
        if not self.save_directory or not os.path.isdir(self.save_directory):
            msg = "Thư mục lưu không hợp lệ. Vui lòng chọn lại."
            self._log(f"[{source}] Ghi thất bại: {msg}", "Record")
            QMessageBox.critical(self, "Lỗi Thư mục Lưu", msg)
            self._update_status("LỖI: Thư mục lưu không hợp lệ.")
            return False

        # --- Generate Filename & Create Writer ---
        full_filepath = self._generate_video_filename() # Generates based on self.loop_count
        if self._create_video_writer(full_filepath):
            # --- Success: Update State & UI ---
            self.is_recording = True
            self.is_paused = False # Ensure not paused when starting a new loop
            status_msg = f"Bắt đầu ghi Loop {self.loop_count}: {os.path.basename(full_filepath)}"
            self._update_status(status_msg)
            self._log(f"{status_msg} (Nguồn: {source})", "Record")

            # Update buttons: Pause & Stop are active, Start is inactive
            # self.btn_start_record.setEnabled(False)
            self.btn_pause_record.setEnabled(True); self.btn_pause_record.setText("Tạm dừng")
            self.btn_stop_save_record.setEnabled(True)

            self._update_status_visuals() # Update status label immediately
            if not self.status_timer.isActive():
                self.status_timer.start(500) # Start/ensure flashing timer is running
            return True # Indicate success
        else:
             # --- Failure ---
             self._log(f"[{source}] Ghi thất bại: Không thể tạo video writer.", "Record")
             self.last_video_filepath = "" # Clear path since file creation failed
             # Update UI to reflect failed recording state
             self.is_recording = False
             # self.btn_start_record.setEnabled(False) # Keep disabled maybe?
             self.btn_pause_record.setEnabled(False)
             self.btn_stop_save_record.setEnabled(False)
             self._update_status("LỖI: Không thể bắt đầu ghi video.")
             self._update_status_visuals()
             return False # Indicate failure


    def _pause_recording(self, source="Manual"):
        """Toggle pause/resume state of recording."""
        if not self.is_recording:
             self._log(f"[{source}] Lệnh Tạm dừng/Tiếp tục bị bỏ qua: Chưa ghi.", "Record")
             return

        self.is_paused = not self.is_paused # Toggle state
        if self.is_paused:
            # self.btn_pause_record.setText("Tiếp tục") # Button text handles both now
            status_msg = f"Đã tạm dừng ghi Loop {self.loop_count}."; log_msg = f"Tạm dừng ghi [{source}] (Loop {self.loop_count})"
        else:
            # self.btn_pause_record.setText("Tạm dừng")
            status_msg = f"Đã tiếp tục ghi Loop {self.loop_count}."; log_msg = f"Tiếp tục ghi [{source}] (Loop {self.loop_count})"

        self._update_status(status_msg); self._log(log_msg, "Record"); self._update_status_visuals()


    def _stop_recording_base(self, action_type, source):
        """Core logic: Stop recording, handle file (Save/Discard), INCREMENT counter, and try to AUTO-RESTART."""
        if not self.is_recording:
             self._log(f"[{source}] Lệnh Dừng ({action_type}) bị bỏ qua: Chưa ghi.", "Record"); return False

        current_loop = self.loop_count # Capture current loop number for logging
        file_path_to_handle = self.last_video_filepath # Get path before clearing/changing it
        file_basename = os.path.basename(file_path_to_handle or f"loop{current_loop}_unknown")
        self._log(f"Yêu cầu Dừng ({action_type}) cho Loop {current_loop}: {file_basename} (Nguồn: {source})", "Record")

        # 1. Immediately update state to prevent further writes/actions on this loop
        self.is_recording = False
        self.is_paused = False
        # Briefly update UI to show stopping (may be overwritten by restart quickly)
        self._update_status(f"Đang dừng Loop {current_loop}...")
        # self.btn_pause_record.setEnabled(False); self.btn_stop_save_record.setEnabled(False) # Disable temporarily
        self._update_status_visuals() # Reflect stop in progress
        QApplication.processEvents() # Process UI update

        # 2. Release the writer
        writer = self.video_writer # Get local ref
        self.video_writer = None # Clear object reference immediately
        file_exists_after_release = False
        if writer and writer.isOpened():
            self._log(f"Đang giải phóng writer cho {file_basename}...", "Record")
            try:
                 writer.release()
                 self._log(f"Writer cho {file_basename} đã giải phóng.", "Record")
                 # Check if file exists *after* release (release is blocking)
                 file_exists_after_release = os.path.exists(file_path_to_handle)
                 if file_exists_after_release:
                     print(f"File verified exists after release: {file_path_to_handle}")
                 else:
                      print(f"Warning: File *does not* exist after release: {file_path_to_handle}")
            except Exception as e:
                self._log(f"LỖI khi giải phóng writer ({file_basename}): {e}", "Record")
                print(f"Error releasing writer ({file_basename}): {e}", file=sys.stderr)
        elif writer: # Writer object existed but wasn't opened (shouldn't happen often)
             self._log(f"Cảnh báo: Writer cho {file_basename} đã đóng trước khi dừng.", "Record")
        else:
             self._log("Cảnh báo: Không tìm thấy đối tượng writer khi dừng.", "Record")


        # 3. Handle the file based on action (Save/Discard)
        final_status_msg = ""
        final_log_msg = ""
        if action_type == "Save":
             if file_path_to_handle and file_exists_after_release:
                 final_status_msg = f"Đã lưu Loop {current_loop}: {file_basename}"
                 final_log_msg = f"Đã dừng & lưu [{source}] Loop {current_loop}: {file_basename}"
                 print(f"Video saved: {file_path_to_handle}") # Console confirmation
             elif file_path_to_handle: # File existed before release but not after? Or path is invalid?
                 final_status_msg = f"LỖI LƯU Loop {current_loop}: {file_basename} (File mất sau release?)"
                 final_log_msg = f"Dừng & LỖI Lưu [{source}] Loop {current_loop}: {file_basename} (Không tìm thấy file sau release)"
                 QMessageBox.warning(self, "Lưu Thất Bại", f"Không tìm thấy file video sau khi đóng:\n{file_path_to_handle}")
             else: # Should not happen if path was generated
                  final_status_msg = f"Lỗi Lưu Loop {current_loop}: Thiếu đường dẫn file"; final_log_msg = f"Dừng & Lưu [{source}]: Lỗi thiếu đường dẫn file?"
        else: # Discard
             final_status_msg = f"Đã dừng & hủy Loop {current_loop}."
             final_log_msg = f"Đã dừng & hủy [{source}] Loop {current_loop}: {file_basename}"
             if file_path_to_handle and os.path.exists(file_path_to_handle): # Check if exists before deleting
                  self._log(f"Đang xóa file bị hủy: {file_basename}", "Record")
                  try:
                      os.remove(file_path_to_handle)
                      self._log(f"Đã xóa file: {file_basename}", "Record")
                  except OSError as e:
                      err_msg = f"LỖI khi xóa file bị hủy ({file_basename}): {e}"
                      self._log(err_msg, "Record")
                      print(err_msg, file=sys.stderr)
                      final_status_msg += " (Lỗi xóa file)"
             self.last_video_filepath = "" # Clear path since it's discarded or errored

        # 4. Log the result of the stop action
        self._log(final_log_msg, "Record")
        self._update_status(final_status_msg)


        # 5. **** Increment Loop Counter ****
        self.loop_count += 1
        self._log(f"Bộ đếm Loop được tăng lên: {self.loop_count}", "System")

        # 6. **** Attempt Auto-Restart ****
        webcam_still_ok = bool(self.webcam_thread and self.webcam_thread.isRunning())
        if webcam_still_ok:
            self._log(f"Webcam vẫn chạy, tự động bắt đầu ghi Loop {self.loop_count}...", "Record")
            # Short delay might help ensure file system is ready or webcam has settled? Optional.
            # self.thread().msleep(50)
            QApplication.processEvents() # Allow UI to process previous status updates
            if self._start_recording("Auto-Restart"):
                 # Success: _start_recording updated UI and status
                 pass
            else:
                 # Failure: _start_recording handled logging/message/UI state for failure
                 self._log(f"LỖI: Không thể tự động khởi động lại ghi hình cho Loop {self.loop_count}.", "Record")
                 self._update_status(f"LỖI Khởi động lại Loop {self.loop_count}. Webcam có thể vẫn chạy.")
                 # Reset buttons to non-recording state as restart failed
                 # self.btn_start_record.setEnabled(False)
                 self.btn_pause_record.setEnabled(False)
                 self.btn_stop_save_record.setEnabled(False)
        else:
             # Cannot restart because webcam stopped/stopping
             self._log("Không tự động ghi lại: Webcam không chạy.", "Record")
             self._update_status(f"Loop {current_loop} đã {action_type.lower()}. Webcam đã dừng.")
             # Ensure buttons are fully disabled as webcam stopped
             # self.btn_start_record.setEnabled(False)
             self.btn_pause_record.setEnabled(False); self.btn_pause_record.setText("Tạm dừng / Tiếp tục")
             self.btn_stop_save_record.setEnabled(False)


        # 7. Final UI state update based on whether restart happened
        self._update_status_visuals()

        # Clear the filename path of the loop that just finished
        # Do this *after* potential deletion and *before* restart generates new path
        self.last_video_filepath = "" if action_type == "Discard" else file_path_to_handle # Keep path on Save?

        return True # Indicate stop action was processed


    def _stop_save_recording(self, source="Manual"):
        """Stop recording, save the file, increment count, and auto-restart."""
        return self._stop_recording_base("Save", source)

    def _stop_discard_recording(self, source="Manual"):
        """Stop recording, discard the file, increment count, and auto-restart."""
        return self._stop_recording_base("Discard", source)

    # --- Manual Button Wrappers ---
    # def _manual_start_recording(self): self._start_recording("Manual") # No longer needed
    def _manual_pause_recording(self): self._pause_recording("Manual")
    def _manual_stop_save_recording(self):
        # This button action triggers the full Save -> Increment -> Restart cycle
        self._stop_save_recording("ManualButton")


    # ================== Application Level Methods ==================

    def _confirm_and_stop_recording(self, message):
         """Ask user whether to Save/Discard/Cancel the CURRENT loop. Returns True if stopped (Save or Discard), False if Cancelled."""
         if not self.is_recording: return True # No recording active, safe to proceed

         self._log(f"Hiển thị hộp thoại xác nhận dừng: '{message[:50]}...'", "System")
         reply = QMessageBox.question(self, f'Xác nhận Dừng Loop {self.loop_count}', message,
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                     QMessageBox.Save) # Default to Save

         if reply == QMessageBox.Save:
             self._log("Người dùng chọn LƯU.", "System")
             # Call base function for Save. It returns True if processed.
             # CRITICAL: It will also try to auto-restart. We might not want that during app close/webcam stop.
             # The auto-restart logic within _stop_recording_base checks if the webcam thread is running.
             # If we are calling this during webcam stop or app close, the thread *should* be stopping or stopped,
             # preventing the unwanted restart.
             return self._stop_recording_base("Save", "ConfirmDialog")
         elif reply == QMessageBox.Discard:
             self._log("Người dùng chọn HỦY BỎ (Discard).", "System")
             # Call base function for Discard. Handles deletion, counter, restart attempt.
             return self._stop_recording_base("Discard", "ConfirmDialog")
         else: # reply == QMessageBox.Cancel
            self._log("Người dùng chọn HỦY (Cancel).", "System")
            return False # User cancelled the parent action (like closing app or stopping webcam)

    def closeEvent(self, event):
        """Handle window close event: confirm recording stop, stop threads."""
        self._log("Yêu cầu đóng ứng dụng...", "System")
        should_close = True

        # --- Confirm Stop Recording if active ---
        if self.is_recording:
            self._log("Ứng dụng đang đóng nhưng ghi hình đang chạy. Xác nhận Lưu/Hủy...", "Record")
            should_close = self._confirm_and_stop_recording(
                f"Ứng dụng đang đóng.\nVideo hiện tại ({os.path.basename(self.last_video_filepath or '...')}) chưa được lưu.\nBạn có muốn lưu lại Loop {self.loop_count} không?"
            )
            # _confirm_and_stop_recording returns True if Save/Discard chosen, False if Cancel chosen.
            # If True, it also called _stop_recording_base which handled the file, incremented count,
            # and *attempted* restart (which should fail here as webcam will be stopped).

        if not should_close:
            self._log("Việc đóng ứng dụng bị hủy bởi người dùng (tại xác nhận ghi hình).", "System")
            event.ignore() # Prevent window from closing
            return

        # --- Proceed with closing ---
        self._update_status("Đang đóng ứng dụng...")
        QApplication.processEvents() # Update UI
        self.status_timer.stop()

        # --- Stop Threads Gracefully ---
        self._log("Bắt đầu dừng các thread worker...", "System")
        # Stop webcam thread FIRST (it handles its own video cleanup via _confirm_and_stop above)
        if self.webcam_thread and self.webcam_thread.isRunning():
             self._log("Yêu cầu dừng thread webcam từ closeEvent...", "Webcam")
             self._stop_webcam() # Calls internal stop sequence
             # Wait briefly for webcam thread to potentially finish after stop() call returns?
             # stop() has a wait(), maybe not needed here. self.webcam_thread.wait(500)

        # Stop serial thread
        if self.serial_thread and self.serial_thread.isRunning():
             self._log("Yêu cầu dừng thread serial từ closeEvent...", "Serial")
             self._disconnect_serial() # Calls internal disconnect sequence
             # Wait? self.serial_thread.wait(500)

        # --- Final Video Writer Check (Safety Net) ---
        # Should have been released by stop sequence, but double check
        if self.video_writer and self.video_writer.isOpened():
             self._log("CẢNH BÁO: Phát hiện video writer vẫn mở trong closeEvent. Đang đóng...", "System")
             try: self.video_writer.release(); print(" -> Released stray writer on exit.")
             except Exception as e: print(f" -> Lỗi đóng writer tồn đọng khi thoát: {e}")
             self.video_writer = None

        self._log("Ứng dụng thoát.", "System")
        print("Exiting application.")
        event.accept() # Allow window to close


# =============================================================================
# == Application Entry Point ==
# =============================================================================
if __name__ == '__main__':
    # Enable High DPI support if needed
    # QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    # QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

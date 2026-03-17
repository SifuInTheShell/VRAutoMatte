"""Light and dark themes for VRAutoMatte UI.

Each theme provides a Qt stylesheet and a color map for
inline styles that can't be set via global stylesheet
(e.g. dynamically toggled labels).
"""

# ── Shared layout constants ──
_SHARED = """
/* ── Shared ── */
* {
    font-family: 'Segoe UI Variable', 'Segoe UI', system-ui;
    font-size: 13px;
}
QGroupBox {
    border-radius: 10px;
    margin-top: 14px;
    padding: 20px 14px 14px 14px;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
}
QLineEdit {
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}
QPushButton {
    border-radius: 6px;
    padding: 7px 18px;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.3px;
}
QPushButton#startButton {
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
    padding: 10px 28px;
    letter-spacing: 0.5px;
}
QPushButton#addBatchButton {
    padding: 7px 14px;
}
QComboBox {
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    min-width: 100px;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
    subcontrol-position: right center;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    border-radius: 4px;
    padding: 4px;
    outline: 0;
}
QCheckBox {
    spacing: 8px;
    font-size: 12px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
}
QSlider::groove:horizontal {
    border: none;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    border: none;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    border-radius: 2px;
}
QProgressBar {
    border-radius: 6px;
    text-align: center;
    height: 24px;
    font-size: 11px;
    font-weight: 600;
}
QProgressBar::chunk {
    border-radius: 5px;
}
QListWidget {
    border-radius: 6px;
    font-size: 12px;
    outline: 0;
}
QListWidget::item {
    padding: 6px 10px;
}
QScrollBar:vertical {
    width: 8px;
    border: none;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    height: 8px;
    border: none;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    border-radius: 4px;
    min-width: 30px;
}
QToolTip {
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 12px;
    max-width: 380px;
}
QSplitter::handle {
    width: 2px;
}
QLabel#statusLabel {
    font-size: 12px;
    font-weight: 500;
}
QLabel#deviceLabel {
    font-size: 11px;
    font-style: italic;
}
QLabel {
    font-size: 12px;
}
"""

# ── Light Theme ──
LIGHT_STYLE = _SHARED + """
QMainWindow, QWidget {
    background-color: #e8eaef;
    color: #2a2c38;
}
QGroupBox {
    background-color: #f2f3f6;
    border: 1px solid #d0d2da;
    color: #4a7888;
}
QGroupBox::title { color: #4a7888; }
QLineEdit {
    background-color: #ffffff;
    border: 1px solid #c8cad4;
    color: #2a2c38;
    selection-background-color: #80c8d8;
}
QLineEdit:focus {
    border-color: #40a0b8;
    background-color: #ffffff;
}
QLineEdit[readOnly="true"] {
    color: #606878;
    background-color: #f0f1f4;
}
QPushButton {
    background-color: #ffffff;
    border: 1px solid #c0c4d0;
    color: #3a3e50;
}
QPushButton:hover {
    background-color: #f0f4fa;
    border-color: #60a0b0;
    color: #2a2e40;
}
QPushButton:pressed {
    background-color: #e4e8f0;
    border-color: #40a0b8;
}
QPushButton:disabled {
    background-color: #ecedf0;
    color: #a0a4b0;
    border-color: #d8dae0;
}
QPushButton#startButton {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #2a8888, stop:1 #2a9898);
    border: 1px solid #30a0a0;
    color: #ffffff;
}
QPushButton#startButton:hover {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #32a0a0, stop:1 #32b0b0);
    border-color: #40b8b8;
}
QPushButton#startButton:pressed { background-color: #248080; }
QPushButton#startButton:disabled {
    background-color: #b0c8c8;
    border-color: #a0b8b8;
    color: #e0e8e8;
}
QPushButton#cancelButton {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #c05050, stop:1 #d06060);
    border: 1px solid #c06060;
    color: #ffffff;
}
QPushButton#cancelButton:hover {
    background-color: #d06868;
    border-color: #e07070;
}
QPushButton#addBatchButton {
    background-color: transparent;
    border: 1px dashed #90a0b0;
    color: #4a8898;
}
QPushButton#addBatchButton:hover {
    border-color: #40a0b8;
    color: #2a7888;
    background-color: #f0f8fa;
}
QPushButton#themeToggle {
    background-color: transparent;
    border: 1px solid #c0c4d0;
    border-radius: 12px;
    padding: 2px 8px;
    font-size: 14px;
    min-width: 28px;
    max-width: 28px;
    min-height: 24px;
}
QPushButton#themeToggle:hover {
    background-color: #f0f4fa;
    border-color: #60a0b0;
}
QComboBox {
    background-color: #ffffff;
    border: 1px solid #c8cad4;
    color: #2a2c38;
}
QComboBox:hover { border-color: #60a0b0; }
QComboBox::down-arrow { border-top: 5px solid #8090a0; }
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #c0c4d0;
    color: #2a2c38;
    selection-background-color: #d0eef4;
    selection-color: #1a5868;
}
QCheckBox { color: #3a3e50; }
QCheckBox::indicator {
    border: 2px solid #a0a8b8;
    background-color: #ffffff;
}
QCheckBox::indicator:hover { border-color: #50a0b0; }
QCheckBox::indicator:checked {
    background-color: #2a9090;
    border-color: #30a8a8;
}
QCheckBox::indicator:checked:hover { background-color: #32a0a0; }
QSlider::groove:horizontal { background: #d0d4de; }
QSlider::handle:horizontal {
    background: qradialgradient(cx:0.5,cy:0.5,radius:0.5,
        fx:0.5,fy:0.5,
        stop:0 #50c8c8, stop:0.7 #30a0a8, stop:1 #2888a0);
}
QSlider::handle:horizontal:hover {
    background: qradialgradient(cx:0.5,cy:0.5,radius:0.5,
        fx:0.5,fy:0.5,
        stop:0 #70e8e8, stop:0.7 #40b0b8, stop:1 #3098a8);
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #2a9090, stop:1 #38a8b0);
}
QProgressBar {
    border: 1px solid #c8cad4;
    background-color: #ffffff;
    color: #2a7888;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #2a8888, stop:0.5 #38a8a8, stop:1 #2a9090);
}
QLabel { color: #505868; }
QLabel#statusLabel { color: #6a7888; }
QLabel#deviceLabel { color: #4a7888; }
QListWidget {
    background-color: #ffffff;
    border: 1px solid #c8cad4;
    color: #3a3e50;
}
QListWidget::item { border-bottom: 1px solid #ecedf0; }
QListWidget::item:selected {
    background-color: #d8f0f4;
    color: #1a5868;
}
QListWidget::item:hover { background-color: #f0f6f8; }
QSplitter::handle { background-color: #d0d4de; }
QScrollBar:vertical { background: #f0f1f4; }
QScrollBar::handle:vertical { background: #c0c4d0; }
QScrollBar::handle:vertical:hover { background: #a0a8b8; }
QScrollBar:horizontal { background: #f0f1f4; }
QScrollBar::handle:horizontal { background: #c0c4d0; }
QToolTip {
    background-color: #2a3040;
    color: #f0f0f8;
    border: 1px solid #4a5868;
}
"""

# ── Dark Theme ──
DARK_STYLE = _SHARED + """
QMainWindow, QWidget {
    background-color: #1e2028;
    color: #e8e8f0;
}
QGroupBox {
    background-color: #262830;
    border: 1px solid #3a3c48;
    color: #8aaaba;
}
QGroupBox::title { color: #8aaaba; }
QLineEdit {
    background-color: #1a1c24;
    border: 1px solid #3a3c48;
    color: #e8e8f0;
    selection-background-color: #3a7a8a;
}
QLineEdit:focus {
    border-color: #50b0c0;
    background-color: #1e2028;
}
QLineEdit[readOnly="true"] {
    color: #a0a0b0;
    background-color: #222430;
}
QPushButton {
    background-color: #2e3040;
    border: 1px solid #464858;
    color: #d0d0e0;
}
QPushButton:hover {
    background-color: #383a50;
    border-color: #5a8a9a;
    color: #f0f0ff;
}
QPushButton:pressed {
    background-color: #262838;
    border-color: #50b0c0;
}
QPushButton:disabled {
    background-color: #222430;
    color: #50505a;
    border-color: #2e3038;
}
QPushButton#startButton {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #265a5a, stop:1 #2a6a6a);
    border: 1px solid #3a9090;
    color: #c0ffff;
}
QPushButton#startButton:hover {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #2e6a6a, stop:1 #328080);
    border-color: #50b0b0;
    color: #e0ffff;
}
QPushButton#startButton:pressed { background-color: #204848; }
QPushButton#startButton:disabled {
    background-color: #1e2830;
    border-color: #2a3a3a;
    color: #4a6a6a;
}
QPushButton#cancelButton {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
        stop:0 #4a2020, stop:1 #5a2828);
    border: 1px solid #8a4040;
    color: #ffc0c0;
}
QPushButton#cancelButton:hover {
    background-color: #5a2828;
    border-color: #aa5050;
    color: #ffe0e0;
}
QPushButton#addBatchButton {
    background-color: transparent;
    border: 1px dashed #4a5a6a;
    color: #7ab0c0;
}
QPushButton#addBatchButton:hover {
    border-color: #50b0c0;
    color: #a0d0e0;
    background-color: #262a38;
}
QPushButton#themeToggle {
    background-color: transparent;
    border: 1px solid #464858;
    border-radius: 12px;
    padding: 2px 8px;
    font-size: 14px;
    min-width: 28px;
    max-width: 28px;
    min-height: 24px;
    color: #d0d0e0;
}
QPushButton#themeToggle:hover {
    background-color: #383a50;
    border-color: #5a8a9a;
}
QComboBox {
    background-color: #1a1c24;
    border: 1px solid #3a3c48;
    color: #e8e8f0;
}
QComboBox:hover { border-color: #5a8a9a; }
QComboBox::down-arrow { border-top: 5px solid #8888a0; }
QComboBox QAbstractItemView {
    background-color: #1e2028;
    border: 1px solid #464858;
    color: #e8e8f0;
    selection-background-color: #2a5060;
    selection-color: #c0ffff;
}
QCheckBox { color: #d0d0e0; }
QCheckBox::indicator {
    border: 2px solid #4a5a68;
    background-color: #1a1c24;
}
QCheckBox::indicator:hover { border-color: #5a9aaa; }
QCheckBox::indicator:checked {
    background-color: #308080;
    border-color: #50b0b0;
}
QCheckBox::indicator:checked:hover { background-color: #3a9090; }
QSlider::groove:horizontal { background: #2a2c38; }
QSlider::handle:horizontal {
    background: qradialgradient(cx:0.5,cy:0.5,radius:0.5,
        fx:0.5,fy:0.5,
        stop:0 #70d8d8, stop:0.7 #4aacb8, stop:1 #3a8a9a);
}
QSlider::handle:horizontal:hover {
    background: qradialgradient(cx:0.5,cy:0.5,radius:0.5,
        fx:0.5,fy:0.5,
        stop:0 #90f0f0, stop:0.7 #60c0ca, stop:1 #4aa0aa);
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #308080, stop:1 #40a0a8);
}
QProgressBar {
    border: 1px solid #3a3c48;
    background-color: #1a1c24;
    color: #a0d0e0;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #286868, stop:0.5 #3a9a9a, stop:1 #287878);
}
QLabel { color: #b0b0c0; }
QLabel#statusLabel { color: #8090a0; }
QLabel#deviceLabel { color: #6a8a9a; }
QListWidget {
    background-color: #1a1c24;
    border: 1px solid #3a3c48;
    color: #d0d0e0;
}
QListWidget::item { border-bottom: 1px solid #262830; }
QListWidget::item:selected {
    background-color: #2a3a48;
    color: #c0ffff;
}
QListWidget::item:hover { background-color: #282a38; }
QSplitter::handle { background-color: #2a2c38; }
QScrollBar:vertical { background: #1a1c24; }
QScrollBar::handle:vertical { background: #3a3c50; }
QScrollBar::handle:vertical:hover { background: #505068; }
QScrollBar:horizontal { background: #1a1c24; }
QScrollBar::handle:horizontal { background: #3a3c50; }
QToolTip {
    background-color: #2a2c38;
    color: #e8e8f0;
    border: 1px solid #4a5868;
}
"""

# ── Inline color maps for dynamic labels ──
# Keys match widget/purpose; values are CSS color strings.

LIGHT_COLORS = {
    "info_label": "#5a8898",
    "sbs_auto": "#2a8878",
    "pov_warning_default": "#b08030",
    "pov_quality": "#2a8868",
    "pov_fast": "#b08030",
    "device_cpu_warn": "#b08030",
    "preview_header": "#4a7888",
    "preview_pane_header": "#5a8898",
    "preview_pane_bg": "#dcdee6",
    "preview_pane_border": "#c0c4d0",
    "preview_perf": "#2a8878",
    "preview_mono": "#607080",
}

DARK_COLORS = {
    "info_label": "#6a9aaa",
    "sbs_auto": "#50b0a0",
    "pov_warning_default": "#d0a050",
    "pov_quality": "#50c0a0",
    "pov_fast": "#d0a050",
    "device_cpu_warn": "#d0a050",
    "preview_header": "#8aaaba",
    "preview_pane_header": "#6a9aaa",
    "preview_pane_bg": "#161820",
    "preview_pane_border": "#2a2c38",
    "preview_perf": "#50c0b0",
    "preview_mono": "#8a9aaa",
}

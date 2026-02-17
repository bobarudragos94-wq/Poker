"""
Screen capture module for reading the PokerStars window in real-time.
Uses mss for fast cross-platform screen capture and can locate the
PokerStars window automatically.
"""

import re
import sys
import logging
from typing import Optional, Tuple

import numpy as np

try:
    import mss
    import mss.tools
except ImportError:
    mss = None

try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)

# Platform-specific window finding
if sys.platform == "win32":
    try:
        import win32gui
        import win32con
        HAS_WIN32 = True
    except ImportError:
        HAS_WIN32 = False
elif sys.platform == "linux":
    try:
        import subprocess
        HAS_XDOTOOL = True
    except ImportError:
        HAS_XDOTOOL = False
else:
    HAS_WIN32 = False
    HAS_XDOTOOL = False


class ScreenCapture:
    """Captures screenshots of the PokerStars window."""

    def __init__(self, window_title: str = "PokerStars"):
        self.window_title = window_title
        self._window_rect: Optional[Tuple[int, int, int, int]] = None
        self._sct = mss.mss() if mss else None

    @classmethod
    def _is_pokerstars_table(cls, title: str) -> bool:
        """Check if a window title matches known PokerStars table patterns.

        PokerStars cash tables usually contain 'PokerStars' in the title,
        but tournament tables use titles like:
          'Tournament #123456789 Table 1 - No Limit Hold'em'
          'Spin & Go #123456 Table 1 - ...'
          'Sit & Go #123456 Table 1 - ...'
        which do NOT contain 'PokerStars'.
        """
        title_lower = title.lower()
        # Tournament table: "Tournament #... Table ..."
        if "tournament" in title_lower and "table" in title_lower:
            return True
        # Spin & Go table
        if "spin" in title_lower and "table" in title_lower:
            return True
        # Sit & Go table
        if "sit" in title_lower and "go" in title_lower and "table" in title_lower:
            return True
        # Generic PokerStars table pattern: contains "Table" followed by a number
        if re.search(r'#\d+.*table\s*\d', title_lower):
            return True
        return False

    @classmethod
    def _is_pokerstars_window(cls, title: str) -> bool:
        """Check if a window title belongs to PokerStars (table or lobby)."""
        return cls._is_pokerstars_table(title)

    def invalidate_window(self):
        """Clear cached window position to force re-detection on next capture."""
        self._window_rect = None
        logger.debug("Window rect invalidated - will re-detect on next capture")

    def find_window(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the PokerStars window and return its (left, top, width, height).
        Returns None if the window is not found.
        """
        if sys.platform == "win32":
            return self._find_window_win32()
        elif sys.platform == "linux":
            return self._find_window_linux()
        else:
            logger.warning("Unsupported platform for window detection: %s", sys.platform)
            return None

    @staticmethod
    def _get_dwm_frame_rect(hwnd) -> Optional[Tuple[int, int, int, int]]:
        """Get the actual visible frame bounds using DWM, excluding invisible shadow borders."""
        try:
            import ctypes
            import ctypes.wintypes
            DWMWA_EXTENDED_FRAME_BOUNDS = 9
            rect = ctypes.wintypes.RECT()
            hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
                hwnd, DWMWA_EXTENDED_FRAME_BOUNDS,
                ctypes.byref(rect), ctypes.sizeof(rect)
            )
            if hr == 0:  # S_OK
                return (rect.left, rect.top,
                        rect.right - rect.left, rect.bottom - rect.top)
        except Exception as e:
            logger.debug("DwmGetWindowAttribute failed: %s", e)
        return None

    @staticmethod
    def _get_client_rect(hwnd) -> Optional[Tuple[int, int, int, int]]:
        """Get the client area bounds in screen coordinates, excluding title bar and borders."""
        try:
            # GetClientRect returns (0, 0, width, height) relative to client area
            client_rect = win32gui.GetClientRect(hwnd)
            # ClientToScreen converts client (0,0) to screen coordinates
            client_origin = win32gui.ClientToScreen(hwnd, (0, 0))
            left = client_origin[0]
            top = client_origin[1]
            width = client_rect[2]
            height = client_rect[3]
            if width > 0 and height > 0:
                return (max(0, left), max(0, top), width, height)
        except Exception as e:
            logger.debug("GetClientRect failed: %s", e)
        return None

    def _find_window_win32(self) -> Optional[Tuple[int, int, int, int]]:
        """Find window on Windows using win32gui.

        Prefers actual table windows over lobby/generic PokerStars windows.
        Uses the client area (excluding title bar) for accurate region mapping.
        """
        if not HAS_WIN32:
            logger.warning("pywin32 not installed - cannot auto-detect window")
            return None

        # Collect table windows and generic PokerStars windows separately
        table_windows = []
        generic_windows = []

        def enum_callback(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return

            is_table = self._is_pokerstars_table(title)
            is_generic = self.window_title.lower() in title.lower()

            if not is_table and not is_generic:
                return

            # Prefer client area (excludes title bar) for accurate region mapping
            rect = ScreenCapture._get_client_rect(hwnd)
            if rect is None:
                # Fallback: DWM frame bounds (excludes shadow but includes title bar)
                rect = ScreenCapture._get_dwm_frame_rect(hwnd)
            if rect is None:
                # Last resort: GetWindowRect with clamping
                wr = win32gui.GetWindowRect(hwnd)
                left, top, right, bottom = wr
                adj_left = max(0, left)
                adj_top = max(0, top)
                rect = (adj_left, adj_top, right - adj_left, bottom - adj_top)

            if is_table:
                table_windows.append((rect, title))
                logger.debug("Found table window: '%s' at %s", title, rect)
            elif is_generic:
                generic_windows.append((rect, title))
                logger.debug("Found generic PokerStars window: '%s' at %s", title, rect)

        win32gui.EnumWindows(enum_callback, None)

        # Prefer table windows over lobby/generic windows
        if table_windows:
            self._window_rect, title = table_windows[0]
            logger.info("Found PokerStars table at %s ('%s')", self._window_rect, title)
            return self._window_rect

        if generic_windows:
            self._window_rect, title = generic_windows[0]
            logger.info("Found PokerStars window at %s ('%s') "
                        "(no table window found - may be lobby)",
                        self._window_rect, title)
            return self._window_rect

        logger.debug("PokerStars window not found")
        return None

    def _find_window_linux(self) -> Optional[Tuple[int, int, int, int]]:
        """Find window on Linux using xdotool/wmctrl."""
        try:
            # Build pattern that matches both regular and tournament tables
            escaped_title = re.escape(self.window_title)
            search_pattern = f"{escaped_title}|Tournament.*Table|Spin.*Go.*Table"
            result = subprocess.run(
                ["xdotool", "search", "--name", search_pattern],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                window_id = result.stdout.strip().split('\n')[0]
                geo_result = subprocess.run(
                    ["xdotool", "getwindowgeometry", "--shell", window_id],
                    capture_output=True, text=True, timeout=5
                )
                if geo_result.returncode == 0:
                    geo = {}
                    for line in geo_result.stdout.strip().split('\n'):
                        if '=' in line:
                            k, v = line.split('=', 1)
                            geo[k] = int(v)
                    # Get window size
                    size_result = subprocess.run(
                        ["xdotool", "getwindowfocus", "getwindowgeometry",
                         "--shell", window_id],
                        capture_output=True, text=True, timeout=5
                    )
                    self._window_rect = (
                        geo.get('X', 0), geo.get('Y', 0),
                        geo.get('WIDTH', 800), geo.get('HEIGHT', 600)
                    )
                    logger.info("Found PokerStars window at %s", self._window_rect)
                    return self._window_rect
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("xdotool not available for window detection")
        return None

    def set_window_rect(self, left: int, top: int, width: int, height: int):
        """Manually set the window rectangle (for calibration)."""
        self._window_rect = (left, top, width, height)
        logger.info("Window rect set to: %s", self._window_rect)

    def capture_full_table(self) -> Optional[np.ndarray]:
        """
        Capture the full PokerStars table as a numpy array (BGR format).
        Returns None if capture fails.
        """
        if not self._sct:
            logger.error("mss not available for screen capture")
            return None

        if not self._window_rect:
            self.find_window()

        if not self._window_rect:
            logger.debug("No window rect available - capturing full screen")
            monitor = self._sct.monitors[1]  # Primary monitor
        else:
            left, top, width, height = self._window_rect
            monitor = {"left": left, "top": top, "width": width, "height": height}

        try:
            screenshot = self._sct.grab(monitor)
            # Convert to numpy array (BGRA -> BGR)
            img = np.array(screenshot)
            return img[:, :, :3]  # Drop alpha channel
        except Exception as e:
            logger.error("Screen capture failed: %s", e)
            return None

    def capture_region(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Capture a specific region relative to the PokerStars window.
        region: (x, y, width, height) relative to window top-left.
        """
        if not self._sct:
            return None

        if not self._window_rect:
            self.find_window()
            if not self._window_rect:
                return None

        win_left, win_top, _, _ = self._window_rect
        rx, ry, rw, rh = region

        monitor = {
            "left": win_left + rx,
            "top": win_top + ry,
            "width": rw,
            "height": rh,
        }

        try:
            screenshot = self._sct.grab(monitor)
            img = np.array(screenshot)
            return img[:, :, :3]
        except Exception as e:
            logger.error("Region capture failed for %s: %s", region, e)
            return None

    def capture_to_pil(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional["Image.Image"]:
        """Capture a region and return as PIL Image (RGB)."""
        if Image is None:
            return None

        if region:
            img = self.capture_region(region)
        else:
            img = self.capture_full_table()

        if img is None:
            return None

        # Convert BGR to RGB
        return Image.fromarray(img[:, :, ::-1])

    def is_window_found(self) -> bool:
        """Check if the PokerStars window has been located."""
        return self._window_rect is not None

    @property
    def window_rect(self) -> Optional[Tuple[int, int, int, int]]:
        return self._window_rect

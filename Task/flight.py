'''
This contains the opencv, line following stuff. This will make the decisions for the robot based on what the camera sees and 
send commands to the controller.
'''
import os
# MUST set before importing cv2 so Qt picks up xcb backend
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

import cv2
import numpy as np
from control import Control
import time
from sensor import Camera
import queue
import math

# REQUIRED INPUT VARIABLE IN CODE
# The array must contain two values. Format: [country1, country2]
# If only one country needs to be visited, the second value will be: 0
Airports = [1, 2]

# ── HSV color ranges for detection (H: 0-179, S: 0-255, V: 0-255) ──────────
COLOR_RANGES = {
    'red':    ([  0,  80, 80], [ 10, 255, 255],   # lower red hue
               [160,  80, 80], [179, 255, 255]),   # upper red hue (wraps)
    'blue':   ([ 90, 80, 80], [130, 255, 255], None, None),
    'green':  ([ 35, 60, 60], [ 85, 255, 255], None, None),
    'yellow': ([ 20, 80, 80], [ 35, 255, 255], None, None),
}

# BGR colors for drawing bounding boxes
DRAW_COLOR = {
    'red':    (0,   0,   255),
    'blue':   (255, 0,   0),
    'green':  (0,   255, 0),
    'yellow': (0,   200, 255),
}

MIN_CONTOUR_AREA = 500   # px² — ignore tiny blobs


class Brain:
    def __init__(self):
        self.control = Control()
        self.camera = Camera()
        self._latest_detections = []
        self._latest_frame = None            # raw BGR frame for line follow
        self._latest_display = None          # annotated frame from process_frame
        self._frame_queue = queue.Queue(maxsize=2)  # main-thread display queue

        # PID state
        self._prev_angle = 0.0
        self._prev_error = 0.0
        self._prev_yr = 0.0
        self._smooth_angle = 0.0
        self._smooth_error = 0.0
        self._t_ctrl = time.time()
        
        self._pid_last_time = None
        self._vy_smooth = 0.0          # smoothed lateral output
        self._prev_vy = 0.0            # for rate limiter

        # AprilTag detector (tag36h11 family used in world)
        self._tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        self._tag_params = cv2.aruco.DetectorParameters()
        self._tag_detector = cv2.aruco.ArucoDetector(self._tag_dict, self._tag_params)
        self._tag_land_area = 20000  # px² — trigger landing only when drone is directly over pad

        # Scan-once tag memory: {tag_id: (country, status, reachables)}
        self._scanned_tags = {}

    # ── frame processing (runs in camera thread) ─────────────────────────────
    def process_frame(self, frame):
        """
        Detect colored objects in the frame, draw enhanced bounding boxes,
        and store detections.
        """
        self._latest_frame = frame          # store raw frame for line_follow()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        display = frame.copy()
        overlay = frame.copy()
        detections = []
        fh, fw = display.shape[:2]

        for color_name, ranges in COLOR_RANGES.items():
            lo1, hi1, lo2, hi2 = ranges

            mask = cv2.inRange(hsv, np.array(lo1), np.array(hi1))
            if lo2 is not None:
                mask |= cv2.inRange(hsv, np.array(lo2), np.array(hi2))

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((5, 5), np.uint8))
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                detections.append({
                    'color': color_name,
                    'bbox':  (x, y, w, h),
                    'center': (cx, cy),
                    'area':  area,
                })

                bgr = DRAW_COLOR[color_name]

                # Semi-transparent filled contour
                cv2.fillPoly(overlay, [cnt], bgr)

                # Corner-bracket style box
                t = max(2, min(w, h) // 6)   # bracket arm length
                lw = 2
                for px, py in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
                    sx = 1 if px == x else -1
                    sy = 1 if py == y else -1
                    cv2.line(display, (px, py), (px + sx*t, py), bgr, lw+1)
                    cv2.line(display, (px, py), (px, py + sy*t), bgr, lw+1)
                # Thin full rect
                cv2.rectangle(display, (x, y), (x+w, y+h), bgr, 1)

                # Label with dark background
                label = f"{color_name}  {area:.0f}px"
                (lw2, lh2), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display, (x, y - lh2 - 10), (x + lw2 + 6, y), (0, 0, 0), -1)
                cv2.putText(display, label, (x + 3, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)

                # Centre dot + crosshair
                cv2.circle(display, (cx, cy), 5, bgr, -1)
                cv2.circle(display, (cx, cy), 10, bgr, 1)

        # Blend semi-transparent overlay
        cv2.addWeighted(overlay, 0.18, display, 0.82, 0, display)

        # Frame-centre crosshair
        cv2.line(display, (fw//2 - 20, fh//2), (fw//2 + 20, fh//2), (220, 220, 220), 1)
        cv2.line(display, (fw//2, fh//2 - 20), (fw//2, fh//2 + 20), (220, 220, 220), 1)
        cv2.circle(display, (fw//2, fh//2), 4, (220, 220, 220), 1)

        # Top-left HUD
        cv2.rectangle(display, (0, 0), (200, 26), (0, 0, 0), -1)
        cv2.putText(display, f"Objects: {len(detections)}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        self._latest_detections = detections
        self._latest_display = display      # save annotated frame for line_follow

    # ── helpers ──────────────────────────────────────────────────────────────
    def get_detections(self):
        """Return the most recently detected objects (thread-safe read)."""
        return list(self._latest_detections)

    def _detect_apriltag(self, frame):
        """Detect AprilTag in frame. Returns (tag_id, cx, cy, area, corners) or None."""
        corners, ids, _ = self._tag_detector.detectMarkers(frame)
        if ids is None or len(ids) == 0:
            return None
        best, best_area = None, 0
        for i, tc in enumerate(corners):
            c = tc[0]
            w = np.linalg.norm(c[0] - c[1])
            h = np.linalg.norm(c[0] - c[3])
            area = w * h
            if area > best_area:
                best_area = area
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                best = (int(ids[i][0]), cx, cy, area, tc)
        return best

    def ema(self, prev, current, alpha):
        return alpha * current + (1.0 - alpha) * prev

    def apply_dead_zone(self, val, dead_zone):
        return 0.0 if abs(val) < dead_zone else val

    def line_follow(self, duration=30, forward_speed=0.15, land_on_tag=False, tag_ignore_secs=0, initial_straight_time=5.0):
        """
        PID line follower using the yellow line detected by the downward camera.
        Uses bottom 40% ROI, separate PD controllers for yaw and lateral velocity,
        EMA smoothing, and dead-zones for high stability.
        Returns tag_id if a large enough tag is spotted, else None.
        """
        print(f"Line following for {duration}s  (forward={forward_speed} m/s, tag_ignore={tag_ignore_secs}s, straight={initial_straight_time}s)")
        fw = 640
        start_time = time.time()
        tag_ignore_until = start_time + tag_ignore_secs
        deadline = start_time + duration
        
        # PID constants — tuned low to prevent spinning
        KP_YAW, KD_YAW = 0.015, 0.002
        KP_LAT, KD_LAT = 0.0004, 0.0001
        
        line_lost_count = 0
        ever_found_line = False

        while time.time() < deadline:
            loop_elapsed = time.time() - start_time
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.02)
                continue

            fh, fw = frame.shape[:2]
            roi = frame[int(fh * 0.6):, :]

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

            roi_y0 = int(fh * 0.6)
            disp = self._latest_display.copy() if self._latest_display is not None else frame.copy()

            cv2.rectangle(disp, (0, roi_y0), (fw - 1, fh - 1), (80, 80, 80), 1)
            cv2.putText(disp, "ROI", (4, roi_y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            vy_cmd = 0.0
            yr_cmd = 0.0
            cur_fwd = forward_speed

            if contours:
                best_cnt = max(contours, key=cv2.contourArea)
                M = cv2.moments(best_cnt)
                if M['m00'] > 500:
                    ever_found_line = True
                    line_lost_count = 0
                    cx = int(M['m10'] / M['m00'])
                    raw_error = cx - fw // 2

                    # Calculate angle
                    [vx_l, vy_l, _x, _y] = cv2.fitLine(best_cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                    vx_l = float(vx_l[0])
                    vy_l = float(vy_l[0])
                    if vy_l > 0:  # Ensure vector points up the image
                        vx_l, vy_l = -vx_l, -vy_l
                    raw_angle_deg = math.degrees(math.atan2(vx_l, -vy_l))

                    # 1. Dead zones — wide to ignore noise from grayscale cam
                    ang_ctrl = self.apply_dead_zone(raw_angle_deg, 5.0)
                    lat_ctrl = self.apply_dead_zone(raw_error, 10.0)

                    # 2. EMA smoothing
                    self._smooth_angle = self.ema(self._smooth_angle, ang_ctrl, 0.40)
                    self._smooth_error = self.ema(self._smooth_error, lat_ctrl, 0.40)

                    # Time delta
                    now = time.time()
                    dt = max(now - self._t_ctrl, 0.01)
                    self._t_ctrl = now

                    # 3. PD for Yaw
                    p_yaw = KP_YAW * self._smooth_angle
                    d_yaw = KD_YAW * (self._smooth_angle - self._prev_angle) / dt
                    self._prev_angle = self._smooth_angle

                    ang_abs = abs(self._smooth_angle)
                    
                    if loop_elapsed < initial_straight_time:
                        # Go straight, no rotating at all initially
                        raw_yr = 0.0
                        yr_cmd = 0.0
                        self._prev_yr = 0.0
                    else:
                        # Introduce yaw bit-by-bit by ramping up the max slew
                        ramp_factor = min(1.0, (loop_elapsed - initial_straight_time) / 3.0)
                        max_yaw = (0.6 if ang_abs > 28.0 else 0.30) * ramp_factor
                        raw_yr = max(-max_yaw, min(max_yaw, p_yaw + d_yaw))
                        
                        base_slew = 1.0 if ang_abs > 28.0 else 0.20
                        slew = base_slew * max(0.1, ramp_factor)
                        yr_cmd = max(self._prev_yr - slew, min(self._prev_yr + slew, raw_yr))
                        self._prev_yr = yr_cmd

                    # 4. PD for Lateral
                    p_lat = KP_LAT * self._smooth_error
                    d_lat = KD_LAT * (self._smooth_error - self._prev_error) / dt
                    self._prev_error = self._smooth_error

                    vy_cmd = max(-0.12, min(0.12, p_lat + d_lat))
                    
                    if loop_elapsed < initial_straight_time:
                        vy_cmd = max(-0.04, min(0.04, vy_cmd)) # Limit side strafing during straight-phase

                    # 5. Bend logic
                    if ang_abs > 28.0 and loop_elapsed >= initial_straight_time:
                        vy_cmd *= 0.4  # Prioritize yaw over lateral on sharp bends
                    
                    if ang_abs > 15.0 and loop_elapsed >= initial_straight_time:
                        scale = max(0.4, 1.0 - (ang_abs - 15.0) / 90.0)
                        cur_fwd = forward_speed * scale

                    # --- DRAW HUD AND VIZ ---
                    roi_tint = np.zeros_like(disp)
                    roi_tint[roi_y0:, :] = cv2.merge([np.zeros_like(mask), mask, np.zeros_like(mask)])
                    cv2.addWeighted(roi_tint, 0.25, disp, 1.0, 0, disp)

                    cv2.line(disp, (cx, roi_y0), (cx, fh), (0, 255, 255), 2)
                    cv2.line(disp, (fw//2, roi_y0), (fw//2, fh), (100, 100, 100), 1)

                    err_color = (0, 200, 255) if abs(raw_error) < 20 else (0, 60, 255)
                    bar_y = roi_y0 + (fh - roi_y0) // 2
                    cv2.arrowedLine(disp, (fw//2, bar_y), (cx, bar_y), err_color, 2, tipLength=0.15)
                    cv2.circle(disp, (cx, bar_y), 5, err_color, -1)

                    hud = [
                        (f"Line  x={cx}  Ang={raw_angle_deg:+.1f}", (0, 255, 255)),
                        (f"Err  {raw_error:+d} px", err_color),
                        (f"Vy={vy_cmd:+.3f}  Yr={yr_cmd:+.3f}", (200, 255, 200))
                    ]
                    for i, (txt, col) in enumerate(hud):
                        yy = 48 + i * 22
                        cv2.rectangle(disp, (0, yy - 16), (280, yy + 4), (0, 0, 0), -1)
                        cv2.putText(disp, txt, (6, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)

                else:
                    line_lost_count += 1
            else:
                line_lost_count += 1
            
            if line_lost_count > 0:
                cv2.rectangle(disp, (0, 40), (160, 68), (0, 0, 180), -1)
                cv2.putText(disp, f"LOST {line_lost_count}", (6, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 255), 2)
                
                if not ever_found_line and loop_elapsed < initial_straight_time:
                    # Still taking off, push forward to find the start of the line
                    cur_fwd = 0.15
                    vy_cmd = 0.0
                    yr_cmd = 0.0
                elif line_lost_count <= 15:
                    # Keep yawing in last known direction briefly, tiny forward creep
                    cur_fwd = 0.03
                    yr_cmd = self._prev_yr
                    vy_cmd = 0.0
                else:
                    cur_fwd = 0.0
                    vy_cmd = 0.0
                    yr_cmd = 0.0
                    self._prev_yr = 0.0
                    self._prev_angle = 0.0
                    self._smooth_angle = 0.0
                    self._t_ctrl = time.time()

            # ── AprilTag detection ──────────────────────────────────────────
            if time.time() >= tag_ignore_until:
                tag = self._detect_apriltag(frame)
            else:
                tag = None

            if tag is not None:
                tid, tcx, tcy, tarea, tcorners = tag
                print(f"[TAG] AprilTag ID={tid}  area={tarea:.0f}  center=({tcx},{tcy})")

                # Draw tag outline + filled corners
                cv2.aruco.drawDetectedMarkers(disp, [tcorners], np.array([[tid]]))
                for pt in tcorners[0].astype(int):
                    cv2.circle(disp, tuple(pt), 5, (0, 255, 0), -1)

                # Centre rings
                cv2.circle(disp, (tcx, tcy), 8,  (0, 255, 0), 2)
                cv2.circle(disp, (tcx, tcy), 16, (0, 255, 0), 1)
                cv2.drawMarker(disp, (tcx, tcy), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 1)

                # Tag label box
                tag_label = f" TAG {tid}  {tarea:.0f}px "
                (tlw, tlh), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(disp, (tcx - tlw//2 - 2, tcy - 36),
                                    (tcx + tlw//2 + 2, tcy - 36 + tlh + 8), (0, 80, 0), -1)
                cv2.putText(disp, tag_label, (tcx - tlw//2, tcy - 36 + tlh),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if tarea >= self._tag_land_area:
                    # Flash border
                    cv2.rectangle(disp, (2, 2), (fw-2, fh-2), (0, 0, 255), 4)
                    cv2.rectangle(disp, (0, fh-32), (fw, fh), (0, 0, 180), -1)
                    cv2.putText(disp, "TAG FOUND", (fw//2 - 80, fh - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    self._push_display(disp)
                    print(f"[TAG] Tag large enough (area={tarea:.0f}) — stopping.")
                    self.control.set_velocity(0, 0, 0)
                    time.sleep(0.3)
                    return tid

            self._push_display(disp)
            # Apply velocity commands based on PID
            self.control.set_velocity(vx=cur_fwd, vy=vy_cmd, vz=0, yaw_rate=yr_cmd)

        # Stop movement when done
        self.control.set_velocity(0, 0, 0)
        print("Line following complete.")
        return None  # timed out, no tag triggered

    def _detect_box_center(self, frame):
        """
        Detect the landing pad by its bright WHITE edge.
        Finds the largest 4-corner quadrilateral from the white contour,
        computes geometric center as average of the 4 corners.
        Returns (cx, cy, pts_4x2, area) or None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # High threshold — only grab the bright white box edge
        _, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        # Sort all contours largest-first
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:
                break   # already sorted, nothing bigger coming
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                sides = [np.linalg.norm(pts[i] - pts[(i+1) % 4])
                         for i in range(4)]
                if max(sides) == 0:
                    continue
                if min(sides) / max(sides) > 0.45:   # roughly square
                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))
                    return (cx, cy, pts, area)
        return None

    def _center_on_box(self, timeout=60.0):
        """
        Lock on the AprilTag / white-edge box center and land immediately.
        Uses precise bidirectional PD control for fine alignment.
        """
        self._box_landed = False
        LOCK_THRESH_X = 15    # px
        LOCK_THRESH_Y = 15    # px
        STABLE_NEED  = 8      # consecutive stable frames
        
        TAG_ALIGN_P_LAT = 0.0025
        TAG_ALIGN_P_FWD = 0.0025
        MAX_VEL = 0.15

        # Target offsets: often want to drift slightly past center
        TARGET_OFFSET_X = 0
        TARGET_OFFSET_Y = 10  # 10 pixels below center (means move forward a bit)

        # Short stop
        self.control.set_velocity(0, 0, 0)
        time.sleep(0.5)

        def _get_error(frame):
            fh, fw = frame.shape[:2]
            target_x = fw // 2 + TARGET_OFFSET_X
            target_y = fh // 2 + TARGET_OFFSET_Y
            
            tag = self._detect_apriltag(frame)
            if tag is not None:
                tid, tcx, tcy, tarea, tcorners = tag
                return tcx - target_x, tcy - target_y, 'tag', (tcx, tcy, tcorners, tid)
            result = self._detect_box_center(frame)
            if result is not None:
                bcx, bcy, pts, _ = result
                return bcx - target_x, bcy - target_y, 'box', (bcx, bcy, pts)
            return None

        # ── Phase 1: center & lock ────────────────────────────────────────
        print("[LAND] Locking onto pad...")
        stable = 0
        deadline = time.time() + 15.0

        while time.time() < deadline:
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.03)
                continue

            fh, fw = frame.shape[:2]
            disp   = frame.copy()
            det    = _get_error(frame)

            if det is not None:
                ex, ey, source, info = det

                # Draw target
                if source == 'tag':
                    tcx, tcy, tcorners, tid = info
                    cv2.aruco.drawDetectedMarkers(disp, [tcorners], np.array([[tid]]))
                    cv2.circle(disp, (tcx, tcy), 10, (0, 255, 0), 2)
                    cv2.drawMarker(disp, (tcx, tcy), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
                else:
                    bcx, bcy, pts = info
                    for i in range(4):
                        cv2.line(disp, tuple(pts[i]), tuple(pts[(i+1) % 4]), (255, 0, 255), 2)
                    cv2.drawMarker(disp, (bcx, bcy), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)

                cv2.drawMarker(disp, (fw // 2, fh // 2), (0, 255, 255), cv2.MARKER_CROSS, 18, 1)
                
                centered = abs(ex) < LOCK_THRESH_X and abs(ey) < LOCK_THRESH_Y
                col = (0, 255, 0) if centered else (0, 180, 255)
                
                cv2.rectangle(disp, (0, 0), (fw, 32), (0, 0, 0), -1)
                cv2.putText(disp, f"[{source}] err=({ex:+d},{ey:+d})  lock={stable}/{STABLE_NEED}",
                            (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

                if centered:
                    stable += 1
                    self.control.set_velocity(0, 0, 0)
                    if stable >= STABLE_NEED:
                        print(f"[LAND] LOCKED [{source}] err=({ex:+d},{ey:+d}) — descending")
                        self._push_display(disp)
                        break
                else:
                    stable = 0
                    # Align commands
                    vx_c = -ey * TAG_ALIGN_P_FWD
                    vy_c = ex * TAG_ALIGN_P_LAT
                    
                    vx_c = float(np.clip(vx_c, -MAX_VEL, MAX_VEL))
                    vy_c = float(np.clip(vy_c, -MAX_VEL, MAX_VEL))
                    
                    if abs(ey) < LOCK_THRESH_Y: vx_c = 0.0
                    if abs(ex) < LOCK_THRESH_X: vy_c = 0.0
                    
                    self.control.set_velocity(vx=vx_c, vy=vy_c, vz=0)
            else:
                stable = 0
                cv2.rectangle(disp, (0, 0), (fw, 32), (0, 0, 0), -1)
                cv2.putText(disp, "LAND: searching...", (6, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 1)
                # Lost target — default to tiny forward creep to find it
                self.control.set_velocity(0.04, 0, 0)

            self._push_display(disp)
            time.sleep(0.04)

        # Full stop — kill all momentum before descending
        self.control.set_velocity(0, 0, 0)
        time.sleep(0.3)

        # ── Phase 2: straight vertical descent — no lateral movement ──────
        print("[LAND] Descending straight down (locked, no corrections)...")
        t_end  = time.time() + timeout
        last_alt = 3.0

        while time.time() < t_end:
            alt_msg = self.control.master.recv_match(
                type='GLOBAL_POSITION_INT', blocking=False)
            if alt_msg:
                last_alt = alt_msg.relative_alt / 1000.0

            if last_alt <= 0.15:
                print(f"[LAND] {last_alt:.2f} m — Touching down...")
                # Hold down for 4 seconds
                t_hold = time.time() + 4.5
                while time.time() < t_hold:
                    self.control.set_velocity(0, 0, 0.4)
                    time.sleep(0.05)
                
                print("[LAND] Grounded wait complete.")
                self.control.set_velocity(0, 0, 0)
                self._box_landed = True
                return

            vz = 0.35 if last_alt > 1.50 else \
                 0.20 if last_alt > 0.80 else \
                 0.10 if last_alt > 0.40 else 0.05

            # Pure vertical — no vx, no vy
            self.control.set_velocity(vx=0, vy=0, vz=vz)

            # Display only
            frame = self._latest_frame
            if frame is not None:
                disp = frame.copy()
                fh, fw = disp.shape[:2]
                cv2.drawMarker(disp, (fw // 2, fh // 2), (0, 255, 255), cv2.MARKER_CROSS, 18, 1)
                cv2.rectangle(disp, (0, 0), (280, 28), (0, 0, 0), -1)
                cv2.putText(disp, f"ALT {last_alt:.2f}m  vz={vz:.2f}  LOCKED",
                            (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)
                self._push_display(disp)

            time.sleep(0.04)

        print("[LAND] Timeout — touching down.")
        # Hold down for 4 seconds
        t_hold = time.time() + 4.5
        while time.time() < t_hold:
            self.control.set_velocity(0, 0, 0.4)
            time.sleep(0.05)
        self.control.set_velocity(0, 0, 0)
        self._box_landed = True

    def _push_display(self, disp):
        """Push annotated frame to the live preview window (main thread only)."""
        try:
            self._frame_queue.put_nowait(disp)
        except queue.Full:
            pass
        try:
            cv2.imshow('Drone Camera', self._frame_queue.get_nowait())
        except queue.Empty:
            pass
        cv2.waitKey(1)

    def _reset_pid(self):
        """Reset PID state (call before a new line-follow phase)."""
        self._prev_angle = 0.0
        self._prev_error = 0.0
        self._prev_yr = 0.0
        self._smooth_angle = 0.0
        self._smooth_error = 0.0
        self._t_ctrl = time.time()
        
        self._pid_last_time  = None
        self._vy_smooth      = 0.0
        self._prev_vy        = 0.0

    # ── main sequence ────────────────────────────────────────────────────────
    def parse_tag(self, tag_id):
        """
        Parses the 3-digit AprilTag ID.
        Digit 1 -> Country Code
        Digit 2 -> Airport Status (1=Safe, 0=Unsafe)
        Digit 3 -> Number of reachable airports
        """
        tag_str = str(tag_id).zfill(3)
        country = int(tag_str[0])
        status = int(tag_str[1])
        reachables = int(tag_str[2])
        return country, status, reachables

    def get_position(self):
        """Get the current GPS position for memory mapping."""
        msg = self.control.master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if msg:
            return (msg.lat / 1e7, msg.lon / 1e7)
        return None

    def align_to_next_path(self, current_tag_id, visited_edges):
        """
        Rotates the drone exactly 90 degrees iteratively to find an unexplored path.
        """
        print(f"[NAV] Making 90-degree turns to find new paths from Airport {current_tag_id}...")
        
        for _ in range(4):  # Check 4 orthogonal directions
            # Stop and look
            self.control.set_velocity(0, 0, 0, yaw_rate=0.0)
            time.sleep(1.0)
            
            frame = self._latest_frame
            if frame is not None:
                # Look at a narrow vertical sliver in the center-bottom
                fh, fw = frame.shape[:2]
                roi = frame[int(fh * 0.6):, int(fw * 0.4):int(fw * 0.6)]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                
                if cv2.countNonZero(mask) > 200:
                    current_yaw = self.control.get_current_yaw()
                    q_yaw = int(round(current_yaw / 90.0) * 90) % 360  # Quantize to 4 directions
                    
                    if (current_tag_id, q_yaw) not in visited_edges:
                        print(f"[NAV] Found an UNEXPLORED path at heading ~{q_yaw}°")
                        visited_edges.add((current_tag_id, q_yaw))
                        return True
                    else:
                        print(f"[NAV] Skipping already visited path at heading ~{q_yaw}°")
            
            # If no suitable line found, snap exactly 90 degrees
            print("[NAV] Turning 90 degrees...")
            self.control.turn_yaw(90)
            time.sleep(0.5)
                        
        print("[NAV] Full 360 completed, no unexplored paths found.")
        self.control.set_velocity(0, 0, 0, yaw_rate=0.0)
        return False

    def start(self):
        """
        Simplified Navigation Sequence:
        1. Take off
        2. Follow line until a tag is found
        3. Scan tag ONCE (skip if already scanned)
        4. Check if it's the target — if yes, land
        5. Otherwise, 90° turn and follow line the other way
        6. Repeat until all target airports found
        """
        print("MAVLink connected. Starting flight sequence...")
        
        self.control.set_mode('GUIDED')
        self.control.force_arm()
        self.control.takeoff(2.2)

        cv2.namedWindow('Drone Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Drone Camera', 640, 480)
        self.camera.start_thread(self.process_frame)
        print("Camera started. Waiting for first frame...")
        while self._latest_frame is None:
            time.sleep(0.05)
            
        targets = [t for t in Airports if t != 0]
        current_target_index = 0
        
        # How many seconds to ignore tags after a turn (avoid immediate re-detect)
        TAG_COOLDOWN = 5
        
        while current_target_index < len(targets):
            target_country = targets[current_target_index]
            print(f"\n[MISSION] Searching for Country {target_country}...")
            
            # Reset PID so line follow starts clean
            self._reset_pid()
            
            # Follow the line until we hit a tag
            tag_id = self.line_follow(
                duration=120,
                forward_speed=0.20,
                land_on_tag=False,
                tag_ignore_secs=TAG_COOLDOWN,
                initial_straight_time=5.0
            )
            
            if tag_id is None:
                print("[MISSION] No tag found within timeout. Landing.")
                break
            
            # ── Check if already scanned ──────────────────────────────────
            if tag_id in self._scanned_tags:
                country, status, reachables = self._scanned_tags[tag_id]
                print(f"[MEMORY] Already scanned tag {tag_id} → Country={country}, Safe={'Yes' if status==1 else 'No'}")
            else:
                # First time seeing this tag — scan and remember
                country, status, reachables = self.parse_tag(tag_id)
                self._scanned_tags[tag_id] = (country, status, reachables)
                print(f"[SCAN] New tag {tag_id} → Country={country}, Safe={'Yes' if status==1 else 'No'}, Reachable={reachables}")
            
            # ── Check if this is our target ────────────────────────────────
            if country == target_country and status == 1:
                print(f"*** FOUND TARGET COUNTRY {target_country} — SAFE! Landing... ***")
                self._center_on_box()  # precise landing
                
                current_target_index += 1
                if current_target_index >= len(targets):
                    print("[MISSION] All targets reached!")
                    break
                else:
                    print(f"[MISSION] Next target: Country {targets[current_target_index]}")
                    self.control.takeoff(2.2)
                    continue
            
            if country == target_country and status == 0:
                print(f"[MISSION] Country {target_country} found but UNSAFE. Continuing search...")
            else:
                print(f"[MISSION] Country {country} ≠ target {target_country}. Continuing...")
            
            # ── 90° turn and continue ─────────────────────────────────────
            print("[NAV] Making 90° turn to follow next path...")
            self.control.set_velocity(0, 0, 0)
            time.sleep(0.5)
            self.control.turn_yaw(90)
            time.sleep(0.5)

        self.control.land()
        print("Flight sequence complete.")
        cv2.destroyAllWindows()

    def __del__(self):
        """Destructor to ensure threads are stopped."""
        self.camera.stop_thread()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    brain = Brain()
    try:
        brain.start()
    except KeyboardInterrupt:
        print("Stopping brain...")
    finally:
        del brain
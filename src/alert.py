import os
import sys
import time
from pathlib import Path
import cv2

# Try to import pygame for cross-platform audio playback
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AUDIO = True
except ImportError:
    PYGAME_AUDIO = False

# Try to import winsound for Windows-specific beeping fallbacks
try:
    import winsound
    WINSOUND_AUDIO = True
except ImportError:
    WINSOUND_AUDIO = False

ALARM_FILE = Path(__file__).resolve().parent.parent / "assets" / "alarm.wav"

class AlarmController:
    def __init__(self):
        self.is_playing = False
        self.pygame_sound = None

        if PYGAME_AUDIO and ALARM_FILE.exists():
            try:
                self.pygame_sound = pygame.mixer.Sound(str(ALARM_FILE))
            except Exception as e:
                print(f"Warning: Failed to load alarm.wav with pygame: {e}")

    def play(self):
        """Starts looping the alarm sound."""
        if self.is_playing:
            return
        
        self.is_playing = True
        print("ALERT // Triggering drowsiness siren!")

        if PYGAME_AUDIO:
            if self.pygame_sound:
                try:
                    self.pygame_sound.play(loops=-1)
                    return
                except Exception as e:
                    print(f"Pygame playback error: {e}")
            else:
                # Generate simple warning sound through pygame's mixer channels if possible
                pass

        if WINSOUND_AUDIO:
            try:
                if ALARM_FILE.exists():
                    winsound.PlaySound(str(ALARM_FILE), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
                else:
                    # Async beep loop in background thread or simple single beep
                    winsound.Beep(2000, 1000)
            except Exception as e:
                print(f"Winsound playback failed: {e}")

    def stop(self):
        """Stops the alarm sound."""
        if not self.is_playing:
            return
        
        self.is_playing = False
        print("ALERT // Stopping drowsiness siren.")

        if PYGAME_AUDIO:
            try:
                pygame.mixer.stop()
            except Exception:
                pass

        if WINSOUND_AUDIO:
            try:
                winsound.PlaySound(None, 0)
            except Exception:
                pass

def draw_warning_overlay(frame, status_text, severity="WARNING"):
    """
    Renders an attention-grabbing warning banner on the screen.
    severity can be: WARNING (amber), CRITICAL (red)
    """
    h, w, _ = frame.shape
    
    # Define color scheme based on alert severity
    if severity == "CRITICAL":
        bg_color = (0, 0, 255)      # Pure Red in BGR
        text_color = (255, 255, 255) # White
    elif severity == "WARNING":
        bg_color = (0, 159, 255)    # Amber/Orange in BGR
        text_color = (255, 255, 255)
    else:
        bg_color = (60, 179, 113)   # Medium Sea Green
        text_color = (255, 255, 255)

    # Render top banner overlay
    banner_height = 55
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), bg_color, -1)
    
    # 25% transparent alpha blend for that premium gloss look
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
    
    # Draw border accent line
    cv2.line(frame, (0, banner_height), (w, banner_height), bg_color, 2)

    # Center-aligned text placement
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    font_thickness = 2
    text_size = cv2.getTextSize(status_text, font, font_scale, font_thickness)[0]
    tx = (w - text_size[0]) // 2
    ty = (banner_height + text_size[1]) // 2

    # Draw text shadow for readability
    cv2.putText(frame, status_text, (tx + 1, ty + 1), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    cv2.putText(frame, status_text, (tx, ty), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

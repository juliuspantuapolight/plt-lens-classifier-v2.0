"""Locate annotations independently of the crops used for classification."""
from dataclasses import replace

import cv2
import numpy as np

from src.tray_geometry import TRAY_CONFIGS, grid_centers


class AnnotationLayout:
    def __init__(self, alignment, slots, tray_type):
        self.alignment = alignment
        self.tray_type = tray_type
        self.gray = cv2.cvtColor(alignment.normalized, cv2.COLOR_BGR2GRAY)
        grid = grid_centers(tray_type, alignment.normalized.shape)
        columns = TRAY_CONFIGS[tray_type]["columns"]
        self.dx = grid[1][2] - grid[0][2]
        self.dy = grid[columns][3] - grid[0][3]
        self.centers = {(row, column): np.array([x, y]) for row, column, x, y in grid}
        self.wells = {}
        self.grid_fitted = False
        self.horizontal = np.array([self.dx, 0.])
        self.vertical = np.array([0., self.dy])
        excluded = set(TRAY_CONFIGS[tray_type].get("excluded", ())) | set(TRAY_CONFIGS[tray_type].get("always_empty", ()))
        if tray_type == "bare":
            for slot in slots:
                key = (slot.row, slot.column)
                if key not in excluded:
                    center = self._well_center(self.centers[key])
                    if center is not None:
                        self.wells[key] = center
        # Fit the visible well grid to account for small camera scale/skew changes.
        if len(self.wells) >= 12:
            keys = list(self.wells)
            if len({row for row, _ in keys}) >= 4 and len({col for _, col in keys}) >= 4:
                source = np.float32([self.centers[key] for key in keys])
                target = np.float32([self.wells[key] for key in keys])
                matrix, inliers = cv2.estimateAffine2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=2.5)
                if matrix is not None and inliers.mean() >= .7:
                    points = np.float32(list(self.centers.values()))
                    shifted = cv2.transform(points[None], matrix)[0]
                    if np.linalg.norm(shifted - points, axis=1).max() < min(self.dx, self.dy) * .25:
                        self.centers = dict(zip(self.centers, shifted))
                        self.grid_fitted = True
                        self.horizontal = matrix[:, :2] @ self.horizontal
                        self.vertical = matrix[:, :2] @ self.vertical

    def _patch(self, center, radius):
        x, y = np.round(center).astype(int)
        left, top = max(0, x - radius), max(0, y - radius)
        return self.gray[top:y + radius + 1, left:x + radius + 1], np.array([left, top])

    def _well_center(self, center):
        patch, origin = self._patch(center, round(min(self.dx, self.dy) * .48))
        contours, _ = cv2.findContours(cv2.Canny(patch, 10, 30), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if not (.45 * self.dx < width < .96 * self.dx and .45 * self.dy < height < .96 * self.dy):
                continue
            if cv2.contourArea(contour) < .65 * width * height:
                continue
            polygon = cv2.approxPolyDP(contour, .04 * cv2.arcLength(contour, True), True)
            if len(polygon) != 4:
                continue
            candidate = np.array([x + (width - 1) / 2, y + (height - 1) / 2]) + origin
            if np.linalg.norm(candidate - center) <= min(self.dx, self.dy) * .25:
                candidates.append((width * height, candidate))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def _aperture_center(self, center):
        # A bounded bright, round aperture is more reliable than the nearest Hough circle.
        radius = round(min(self.dx, self.dy) * (.28 if self.tray_type == "packaged" else .24))
        patch, origin = self._patch(center, radius)
        blurred = cv2.GaussianBlur(patch, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area, perimeter = cv2.contourArea(contour), cv2.arcLength(contour, True)
            if x == 0 or y == 0 or x + width >= patch.shape[1] or y + height >= patch.shape[0]:
                continue
            if not (.75 < width / height < 1.33 and 40 < area < 1000 and perimeter > 0):
                continue
            if 4 * np.pi * area / perimeter ** 2 < .75:
                continue
            moments = cv2.moments(contour)
            candidate = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]) + origin
            if np.linalg.norm(candidate - center) < min(self.dx, self.dy) * .2:
                candidates.append((area, candidate))
        return max(candidates, key=lambda item: item[0])[1] if candidates else center

    def locate(self, slot, label):
        if label == "Excluded":
            return slot
        key = (slot.row, slot.column)
        center = self.centers[key]
        if label == "TLens":
            center = self._aperture_center(center)
        elif not self.grid_fitted and key in self.wells:
            center = self.wells[key]
        original = cv2.perspectiveTransform(np.float32([[center]]), self.alignment.inverse_matrix)[0, 0]
        return replace(slot, normalized_center=tuple(center), original_center=tuple(float(v) for v in original))

    def draw(self, image, slot, label):
        if label == "Excluded":
            return
        x, y = slot.normalized_center
        if label == "TLens":
            radius = min(self.dx, self.dy) * .15
            angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            points = np.column_stack((x + radius * np.cos(angles), y + radius * np.sin(angles)))
            color = (0, 255, 0)
        else:
            center = np.array([x, y])
            horizontal, vertical = self.horizontal / 2, self.vertical / 2
            points = np.array([center - horizontal - vertical, center + horizontal - vertical,
                               center + horizontal + vertical, center - horizontal + vertical])
            color = (0, 0, 255) if label == "Empty" else (0, 255, 255)
        projected = cv2.perspectiveTransform(points.astype(np.float32)[None], self.alignment.inverse_matrix)[0]
        scale = min(image.shape[:2]) / min(self.alignment.normalized.shape[:2])
        thickness = max(1, round(min(self.dx, self.dy) * .055 * scale))
        cv2.polylines(image, [np.round(projected).astype(np.int32)], True, color, thickness, cv2.LINE_AA)

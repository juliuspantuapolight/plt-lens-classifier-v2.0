from pathlib import Path

import cv2
import numpy as np
import pytest

from src.inspection_io import inspection_output_paths
from src.sticker_qr import lot_folder_name, lot_from_value, read_sticker_qr
from src.tray_geometry import align_tray


@pytest.mark.parametrize('value,expected', [
    ('A501B-2633020-003', '2633020'), ('singleword', 'singleword'),
    ('some-lot', 'some-lot'), ('A--003', 'A--003'), ('  custom name  ', 'custom name'),
])
def test_extract_lot(value, expected):
    assert lot_from_value(value) == expected


@pytest.mark.parametrize('value', ['../outside', '/absolute', '..', 'CON', 'a\\b', 'a:b'])
def test_lot_folder_is_one_portable_component(value):
    name = lot_folder_name(value)
    assert '/' not in name and '\\' not in name and ':' not in name
    assert name not in {'.', '..', 'CON'}


def test_outputs_share_lot_folder():
    jpg, report = inspection_output_paths(Path('outputs'), Path('tray.jpg'), '2633020')
    assert jpg == Path('outputs/2633020/tray_inspection.jpg')
    assert report == jpg.with_suffix('.json')


@pytest.mark.parametrize('tray_type', ['packaged', 'bare'])
def test_decode_sticker_qr_and_ignore_codes_outside_sticker(tray_type):
    image = np.full((1600, 1600, 3), 255, np.uint8)
    encoder = cv2.QRCodeEncoder_create()
    code = encoder.encode('A501B-2633020-003')
    code = cv2.resize(code, (120, 120), interpolation=cv2.INTER_NEAREST)
    image[180:300, 120:240] = cv2.cvtColor(code, cv2.COLOR_GRAY2BGR)
    corners = np.float32([[0, 0], [1599, 0], [1599, 1599], [0, 1599]])
    result = read_sticker_qr(image, align_tray(image, corners), tray_type)
    assert result == {'qr_value': 'A501B-2633020-003', 'qr_status': 'decoded', 'lot_number': '2633020'}
    image[180:300, 120:240] = 255
    image[800:920, 800:920] = cv2.cvtColor(code, cv2.COLOR_GRAY2BGR)
    result = read_sticker_qr(image, align_tray(image, corners), tray_type)
    assert result['qr_status'] == 'unreadable'
    assert result['lot_number'] is None

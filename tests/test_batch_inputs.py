from pathlib import Path

from src.inspection_io import collect_input_images


def test_folder_inputs_include_nested_images_and_skip_generated_files(tmp_path):
    names = ['tray.JPG', 'nested/another.png', 'nested/photo.TIFF', 'notes.txt',
             'tray_annotated.jpg', 'nested/another-annotated.PNG', 'tray_inspection.jpg',
             'nested/tray-INSPECTION.jpg', 'results/2633020/other.jpg', 'fake.jpg/child.txt']
    for name in names:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    found = collect_input_images(tmp_path, tmp_path / 'results')
    assert {path.relative_to(tmp_path).as_posix() for path in found} == {
        'tray.JPG', 'nested/another.png', 'nested/photo.TIFF',
    }


def test_empty_folder_has_no_images(tmp_path):
    assert collect_input_images(tmp_path, tmp_path / 'results') == []

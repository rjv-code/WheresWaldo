import pytest

from app.detector import load_detector_module


@pytest.mark.parametrize(
    "image_size, tile_size, stride, expected",
    [
        (100, 512, 448, [0]),
        (512, 512, 448, [0]),
        (1200, 512, 448, [0, 448, 688]),
        (1800, 512, 448, [0, 448, 896, 1288])
    ]
)
def test_tile_starts(image_size, tile_size, stride, expected):
    """Checks that tile positions cover the image dimension"""
    detector = load_detector_module()

    result = detector.tile_starts(image_size, tile_size, stride)

    assert result == expected


def test_tile_starts_cover_edges():
    """Checks that no gaps remain between image tiles"""
    detector = load_detector_module()

    positions = detector.tile_starts(1800, 512, 448)

    assert positions[0] == 0
    assert positions[-1] + 512 == 1800

    for current, following in zip(positions, positions[1:]):
        assert following - current <= 512

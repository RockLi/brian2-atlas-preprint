from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'validation'),str(ROOT/'experiments'),str(ROOT/'python')]
import flywire_mnist_demo as demo


def test_normalization_centers_preserves_aspect_and_is_translation_invariant():
    image=np.zeros((28,28),np.uint8);image[2:12,3:8]=255
    moved=np.zeros_like(image);moved[15:25,18:23]=255
    original=image.copy();a=demo.normalize_digit(image);b=demo.normalize_digit(moved)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(image,original)
    cells=np.argwhere(a>0)
    np.testing.assert_array_equal(cells.max(0)-cells.min(0)+1,[20,10])
    yy,xx=np.indices(a.shape)
    assert abs((yy*a).sum()/a.sum()-13.5)<=.5
    assert abs((xx*a).sum()/a.sum()-13.5)<=.5
    assert a.dtype==np.uint8


def test_blank_and_single_pixel_preprocessing_are_finite():
    blank=np.zeros((28,28),np.uint8)
    np.testing.assert_array_equal(demo.normalize_digit(blank),blank)
    image=blank.copy();image[0,27]=255
    result=demo.normalize_digit(image)
    assert result.sum()==20*20*255
    assert np.isfinite(result).all()


def test_request_rejects_labels_noninteger_values_and_wrong_dimensions():
    pixels=[0]*784
    for data in [dict(pixels=pixels,label=3),dict(pixels=pixels,normalize='false'),
                 dict(pixels=pixels[:-1]),dict(pixels=[256]+pixels[1:]),
                 dict(pixels=[True]+pixels[1:]),dict(pixels=[.5]+pixels[1:])]:
        with pytest.raises(ValueError):demo.parse_request(data)
    original,result,normalize=demo.parse_request(dict(pixels=list(range(256))*3+list(range(16)),normalize=False))
    assert normalize is False
    np.testing.assert_array_equal(original,result)
    result[0,0]=255
    assert original[0,0]==0


def test_only_matching_loopback_origin_can_submit():
    assert demo.local_request({'Host':'127.0.0.1:8765'})
    assert demo.local_request({'Host':'localhost:9999','Origin':'http://localhost:9999'},True)
    for headers in [dict(Host='127.0.0.1:8765'),
                    dict(Host='127.0.0.1:8765',Origin='https://example.com'),
                    dict(Host='example.com:8765',Origin='http://example.com:8765'),
                    dict(Host='user@localhost:8765',Origin='http://user@localhost:8765'),
                    dict(Host='localhost:invalid',Origin='http://localhost:invalid')]:
        assert not demo.local_request(headers,True)

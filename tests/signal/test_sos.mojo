"""Tests for the second-order-section family in `numax.signal.design`:
`OUTPUT_SOS`/`OUTPUT_ZPK` on the designs, `tf2zpk`, `zpk2sos`, `tf2sos`
and `sos2tf`.

A design's sections against SciPy's `output="sos"` exactly (Butterworth
at order 8, an elliptic bandpass), which pins the pairing and the section
order; `OUTPUT_ZPK` through `zpk2sos` and through `zpk2tf` agreeing with
the other two outputs; `tf2zpk` against SciPy on a filter with distinct
roots; `tf2sos` by its round trip through `sos2tf`, since a design's
repeated zeros at `-1` come out of any root finder only to `eps^(1/m)`;
and `sos2tf` against SciPy.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from numax.signal import (
    OUTPUT_SOS,
    OUTPUT_ZPK,
    butter,
    cheby1,
    cheby2,
    ellip,
    sos2tf,
    tf2sos,
    tf2zpk,
    zpk2sos,
    zpk2tf,
)

comptime f64 = DType.float64


def _close(got: List[Float64], want: List[Float64], tol: Float64) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=tol)


def test_design_sections_match_scipy() raises:
    # scipy: signal.butter(8, 0.2, output="sos")
    var b = butter[f64, 8, output=OUTPUT_SOS](0.2)
    assert_equal(b.dim_at(0), 4)
    _close(
        b.to_host(),
        [
            2.3959644103776166e-05,
            4.791928820755233e-05,
            2.3959644103776166e-05,
            1.0,
            -1.0263514742610553,
            0.26864019099379005,
            1.0,
            2.0,
            1.0,
            1.0,
            -1.0868584613628944,
            0.343430940165366,
            1.0,
            2.0,
            1.0,
            1.0,
            -1.2197253651240232,
            0.5076634651740437,
            1.0,
            2.0,
            1.0,
            1.0,
            -1.4515795942478362,
            0.794251053241888,
        ],
        1e-14,
    )
    # scipy: signal.ellip(3, 1, 40, [0.2, 0.5], btype="bandpass", output="sos")
    var e = ellip[f64, 3, output=OUTPUT_SOS](1.0, 40.0, (0.2, 0.5))
    _close(
        e.to_host(),
        [
            0.05548085891202059,
            0.0,
            -0.05548085891202059,
            1.0,
            -0.8043979537162872,
            0.5787198745704308,
            1.0,
            1.2143714666794767,
            1.0000000000000004,
            1.0,
            -0.015330242967220251,
            0.7916440354817714,
            1.0,
            -1.899383498952712,
            0.9999999999999997,
            1.0,
            -1.5082637207837977,
            0.8712016531202087,
        ],
        1e-10,
    )


def test_zpk_output_agrees_with_the_other_two() raises:
    var zpk = cheby1[f64, 5, output=OUTPUT_ZPK](0.5, 0.3, "highpass")
    assert_equal(len(zpk.poles_re), 5)
    var direct = cheby1[f64, 5, output=OUTPUT_SOS](
        0.5, 0.3, "highpass"
    ).to_host()
    var paired = zpk2sos[f64, 3](zpk).to_host()
    _close(paired, direct, 1e-14)
    var tf = cheby1[f64, 5](0.5, 0.3, "highpass")
    var expanded = zpk2tf[f64, 5, 5](
        zpk.zeros_re, zpk.zeros_im, zpk.poles_re, zpk.poles_im, zpk.gain
    )
    _close(expanded.b.to_host(), tf.b.to_host(), 1e-14)
    _close(expanded.a.to_host(), tf.a.to_host(), 1e-14)


def test_tf2zpk_matches_scipy() raises:
    # scipy: signal.tf2zpk(*signal.cheby2(4, 40, 0.3)), roots sorted
    var tf = cheby2[f64, 4](40.0, 0.3)
    var zpk = tf2zpk(tf.b, tf.a)
    assert_almost_equal(zpk.gain, 0.01826742402013966, atol=1e-14)
    var want_z_re: List[Float64] = [
        -0.27869968931358496,
        -0.27869968931358496,
        0.5335550132127799,
        0.5335550132127799,
    ]
    var want_z_im: List[Float64] = [
        -0.9603783021166779,
        0.9603783021166779,
        -0.8457653621871205,
        0.8457653621871205,
    ]
    var want_p_re: List[Float64] = [
        0.5759835984794681,
        0.5759835984794681,
        0.7523292560337906,
        0.7523292560337906,
    ]
    var want_p_im: List[Float64] = [
        -0.15381401055354058,
        0.15381401055354058,
        -0.3909923050713534,
        0.3909923050713534,
    ]
    _check_roots(zpk.zeros_re, zpk.zeros_im, want_z_re, want_z_im)
    _check_roots(zpk.poles_re, zpk.poles_im, want_p_re, want_p_im)


def _check_roots(
    re: List[Float64],
    im: List[Float64],
    want_re: List[Float64],
    want_im: List[Float64],
) raises:
    """Every wanted root is matched by one computed root, to `1e-10`."""
    assert_equal(len(re), len(want_re))
    for i in range(len(want_re)):
        var found = False
        for j in range(len(re)):
            if (
                abs(re[j] - want_re[i]) < 1e-10
                and abs(im[j] - want_im[i]) < 1e-10
            ):
                found = True
        assert_equal(found, True)


def test_tf2sos_round_trips_through_sos2tf() raises:
    var tf = butter[f64, 5](0.3)
    var sos = tf2sos(tf)
    assert_equal(sos.dim_at(0), 3)
    var back = sos2tf(sos)
    var b = back.b.to_host()
    var a = back.a.to_host()
    var tb = tf.b.to_host()
    var ta = tf.a.to_host()
    # Order 5 in 3 sections is order 6 with a trailing zero coefficient.
    for i in range(6):
        assert_almost_equal(b[i], tb[i], atol=1e-10)
        assert_almost_equal(a[i], ta[i], atol=1e-10)
    assert_almost_equal(b[6], 0.0, atol=1e-10)
    assert_almost_equal(a[6], 0.0, atol=1e-10)


def test_sos2tf_matches_scipy() raises:
    # scipy: signal.sos2tf(signal.butter(8, 0.2, output="sos"))
    var tf = sos2tf(butter[f64, 8, output=OUTPUT_SOS](0.2))
    _close(
        tf.b.to_host(),
        [
            2.3959644103776166e-05,
            0.00019167715283020933,
            0.0006708700349057327,
            0.0013417400698114653,
            0.0016771750872643315,
            0.0013417400698114653,
            0.0006708700349057327,
            0.00019167715283020933,
            2.3959644103776166e-05,
        ],
        1e-14,
    )
    _close(
        tf.a.to_host(),
        [
            1.0,
            -4.78451489499581,
            10.445041065534665,
            -13.457719890241556,
            11.129331039163977,
            -6.025260397297651,
            2.079273803011877,
            -0.41721715698978223,
            0.03720010070484524,
        ],
        1e-12,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

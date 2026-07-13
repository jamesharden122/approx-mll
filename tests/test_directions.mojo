from optimizers.directions import DirMethod, DirectionMat
from std.testing import assert_equal, assert_true, TestSuite


def test_directions_gaussian() raises:
    var method = DirMethod.Gaussian()
    assert_equal(method.raw, Int32(0))

    var directions = DirectionMat[8](method)
    var saw_nonzero = False
    for i in range(64):
        var value = directions.p_mat[i]
        assert_true(value == value, msg="Gaussian direction must not be NaN")
        if value != 0.0:
            saw_nonzero = True
    assert_true(saw_nonzero, msg="Gaussian matrix must contain a nonzero value")


def test_directions_identity() raises:
    var method = DirMethod.Identity()
    assert_equal(method.raw, Int32(1))

    var directions = DirectionMat[8](method)
    for row in range(8):
        for column in range(8):
            var expected = 1.0 if row == column else 0.0
            assert_equal(directions.p_mat[row * 8 + column], expected)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

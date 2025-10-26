from optimizers.directions import DirMethod, DirectionMat

fn fail(msg: String):
    print("FAIL:", msg)
    var z: Int = 0
    var crash: Int = 1 // z  # force a runtime error to mark test failure

fn test_directions_gaussian():
    var dm = DirMethod.Gaussian()
    if dm.raw != 0:
        fail("Gaussian tag should be 0")
    var dmat = DirectionMat[8](dm)
    print("test_directions_gaussian: ok; tag=", dm.raw)

fn test_directions_identity():
    var dm = DirMethod.Identity()
    if dm.raw != 1:
        fail("Identity tag should be 1")
    var dmat = DirectionMat[8](dm)
    print("test_directions_identity: ok; tag=", dm.raw)

fn main():
    test_directions_gaussian()
    test_directions_identity()

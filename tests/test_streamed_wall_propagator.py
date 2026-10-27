import numpy as np
from latcoding.pyquda.utils.streamed_wall_propagator import wall_phase


def test_distributed_checkerboard_phase_matches_independent_coordinates():
    global_size=(12,8,10,18);local_size=(6,4,5,9);q=(-4,-2,3)
    for gx,gy,gz in [(0,0,0),(1,1,1)]:
        for t in [0,3,8]:
            actual=wall_phase(local_size,global_size,(gx,gy,gz,0),q,t)
            for eo,z,y,xh in np.ndindex(actual.shape):
                x=next(v for v in [2*xh,2*xh+1] if (v+y+z+t)%2==eo)
                coords=np.array([x+gx*6,y+gy*4,z+gz*5])
                expected=np.exp(2j*np.pi*np.dot(q,coords/np.array(global_size[:3])))
                np.testing.assert_allclose(actual[eo,z,y,xh],expected,atol=1e-14)


def test_wall_source_sink_phases_select_correct_mode_and_volume():
    sizes=(12,8,10,18);q=(-4,-2,3);time=7
    phase=wall_phase(sizes,sizes,(0,0,0,0),q,time)
    # S column equal to its wall source has identity projection. Opposite
    # sign or another transverse mode must not pass that normalization.
    volume=np.prod(sizes[:3]);rhs=phase.conj()
    np.testing.assert_allclose(np.sum(phase*rhs)/volume,1,atol=1e-14)
    other=wall_phase(sizes,sizes,(0,0,0,0),(-4,-1,3),time)
    np.testing.assert_allclose(np.sum(other*rhs)/volume,0,atol=1e-14)

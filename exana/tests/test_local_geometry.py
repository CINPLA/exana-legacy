import numpy as np


def _gaussian(x,y,xc,yc,s):
    return np.exp(-0.5*(((x-xc)/s)**2 + ((y-yc)/s)**2))

def _example_rate_map(sigma=0.05*np.ones(7), spacing=0.3, amplitude=np.ones(7)):
    x = np.linspace(0,1,50)
    y = np.linspace(0,1,50)
    x,y = np.meshgrid(x,y)

    p0 = np.array((0.5,0.5))
    pos = [p0]

    angles = np.linspace(0,2*np.pi,7)[:-1]

    rate_map = np.zeros_like(x)
    rate_map += _gaussian(x,y,*p0,sigma[0])

    for i,a in enumerate(angles):
        p = p0 + [spacing*f(a) for f in [np.cos, np.sin]]
        rate_map += amplitude[i]*_gaussian(x,y,*p,sigma[i])
        pos.append(p)
    return rate_map, np.array(pos)

def test_separate_fields():
    from exana.tracking.fields import separate_fields
    rate_map, _ = _example_rate_map()
    f, nf = separate_fields(rate_map)
    assert nf == 7
    assert np.max(f) == 7
    assert f.shape == (50, 50)

def test_find_avg_dist():
    from exana.tracking.fields import find_avg_dist
    rate_map, _ = _example_rate_map()
    avg_dist = find_avg_dist(rate_map)
    target = 0.3
    # gut feeling value:
    tol = 0.05 

    error = abs(target-avg_dist)/target 
    assert error < tol

    rate_map = np.random.rand(50, 50)
    # running with bad data should give nan maybe?
    # avg_dist = find_avg_dist(rate_map)
    # assert np.isnan(avg_dist)

def test_fit_hex():
    from exana.tracking.fields import fit_hex
    dpos = (0.1,0.1)
    pos = np.array((0.5,0.5)) - dpos
    bump_centers = [pos]
    spacing = 0.3
    a0 = np.pi/3
    
    for a in np.linspace(0,2*np.pi,7)[:-1]:
        bump_centers.append(pos+[spacing*f(a+a0) for f in[np.cos,np.sin]])

    disp1, orient1 = fit_hex(bump_centers, spacing, method = 'best')
    disp2, orient2 = fit_hex(bump_centers, spacing, method = 'closest')

    orientation = (a0*180/np.pi)%60
    displacement = np.linalg.norm(dpos)

    tol = 1e-14

    assert np.abs(displacement - disp1)/displacement < tol
    assert np.abs(orientation - orient1%60)/orientation < tol
    assert np.abs(displacement - disp2)/displacement < tol
    assert np.abs(orientation - orient2%60)/orientation < tol

def test_calculate_grid_geometry():
    from exana.tracking.fields import calculate_grid_geometry

    spacing = 0.3
    import pylab as plt
    rate_map, real_centers = _example_rate_map(spacing=spacing)
    
    bc, avg_dist, displacement, orientation = calculate_grid_geometry(rate_map, plot_fields=True)

    plt.show()
    print(bc)
    print(real_centers)






if __name__ == "__main__":
    # test_separate_fields()
    # test_find_avg_dist()
    # test_fit_hex()
    test_calculate_grid_geometry()
    

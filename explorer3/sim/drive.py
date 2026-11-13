"""Same Tank reverse-diagonal convention as the accepted Explorer controls."""
def command(profile, inputs):
    keys={part for key in inputs for part in key.split('_')}
    forward=int('up' in keys)-int('down' in keys)
    left=int('left' in keys)-int('right' in keys)
    if profile=='tank_dock_sim':
        return [.2*forward,.5*left*(-1 if forward<0 else 1)]
    turn=int('turnleft' in keys)-int('turnright' in keys)
    return [.25*forward,.2*left,.6*turn]

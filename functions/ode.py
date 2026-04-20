
def ode_simple_ms_sys(time_step, x, u):
    """ 
    Calculate new-position of bead in a both-axes based upon NLM.
    
    Args:
        time_step (float):
        x (list): Final position and velocity as [Position_X, Position_Y, Velocity_X, Velocity_Y]
        u (list): Acceleration in [X,Y] directions.
    
    Returns:
        list: Final position and velocity as [Position_X_final, Position_Y_final, Velocity_X_final, Velocity_Y_final]
    """
    x_new = [0, 0, 0, 0]
    # First x-dimension
    x_new[0] = time_step * x[2] + x[0]
    x_new[2] = x[2] + time_step * u[0]

    # Now y-dimension
    x_new[1] = time_step * x[3] + x[1]
    x_new[3] = x[3] + time_step * u[1]

    return x_new

def ode(x0,xd0,F,bv,dt):
    """
    Calculate new-position of bead in a certain-axis only.
    
    Args:
        x0  (float): Initial distance in the axis (can be x or y).
        xd0 (float): Initial velocity in the axis.
        F   (float): Force on the bead (with sign)
        bv  (float): Stoke's coefficient
        dt  (float): Time-step
    
    Returns:
        float: Final position of the bead in the axis.
        float: Final velocity of the bead in the axis. 
        float: The damping force on the bead.
    """
    v = F/bv
    Fdamp = -1*bv*xd0
    x = x0 + v*dt
    return x,v,Fdamp
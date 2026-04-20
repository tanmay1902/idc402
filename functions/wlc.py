import numpy as np

    
def dna_spring(z,basepair):
    """
    Function to calculate the force from DNA-strands due to extension.
    This is according to the force-extension behaviour from Worm-Like Chain.
    
    Args:
		z (float): Extension of DNA-strand
		basepair: The base-pair of the DNA-strand
	
 	Returns:
		float: The force from the DNA-strand.
    """
    P = 4e-9 #Peristence length
    
    # Although L = 0.63*base-pair nm, but we have already scaled bp to order 10^3, so we here directly multiply by 1e-6.
    L = 0.63 * basepair * (1e-9) #Contour Length
    kbT = 1.38e-23 * 300 #kbT value
    
    #Calculate force from WLC (approximation with 15% relative error)
    force = -1 * (kbT/P)*( (1/4) * ( (1 - (z/L))**(-2) ) - (1/4) + (z/L) )
    
    return force
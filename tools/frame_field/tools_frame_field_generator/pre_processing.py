def map_cross_vectors_to_reference_vector(self, angle_rad):
        pi        = torch.tensor(math.pi)

        if angle_rad <0:
            angle_rad = angle_rad + 2*pi
        angle     = angle_rad % (pi/2)
        angles    = torch.tensor([angle,angle + (pi/2),angle+ (pi),angle + (3/2*pi)])
        ref_vec_of_cross          = 4*torch.min(angles)
        return ref_vec_of_cross


